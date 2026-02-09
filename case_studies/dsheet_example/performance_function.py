"""
Performance function for D-Sheet piling reliability analysis.

Limit state function:
    g(x, t) = M_capacity(t) - M_demand(x, t)

where:
    - M_capacity decreases with corrosion (wall thickness loss)
    - M_demand is predicted by MLP surrogate with degraded stiffness
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import json

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from scipy import stats

from src.performance import BasePerformance


class FragilitySurfaceIndex:
    """
    Index for a collection of fragility curves keyed by moment_survived.

    Stores each fragility curve as a separate file, loads on-demand.
    Interpolates between adjacent curves when exact moment not in grid.

    Directory structure:
        fragility_surface/
            manifest.json       # Index with available moment values + metadata
            moment_100.0.npz    # FragilityCurve for moment_survived=100.0
            moment_150.0.npz    # FragilityCurve for moment_survived=150.0
            ...

    Usage:
        # Build and save (expensive, done once)
        index = performance.build_fragility_surface(x, n_moments=50)
        index.save(path / "fragility_surface")

        # Load and query (fast, loads only needed curves)
        index = FragilitySurfaceIndex.load(path / "fragility_surface")
        pf = index.pf_at(corrosion_ratio=0.3, moment_survived=120.0)
    """

    def __init__(
        self,
        directory: Optional[Path] = None,
        corrosion_ratios: Optional[NDArray] = None,
        moment_survived_values: Optional[NDArray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize index.

        Args:
            directory: Directory containing fragility curve files.
            corrosion_ratios: Shared CR grid for all curves.
            moment_survived_values: Available moment thresholds.
            metadata: Additional metadata.
        """
        self.directory = Path(directory) if directory else None
        self.corrosion_ratios = corrosion_ratios
        self.moment_survived_values = moment_survived_values
        self.metadata = metadata or {}

        # Cache for loaded curves: moment_value -> FragilityCurve
        self._cache: Dict[float, "FragilityCurve"] = {}

        # In-memory curves (used during building, before save)
        self._curves: Dict[float, "FragilityCurve"] = {}

    def add_curve(self, moment_survived: float, curve: "FragilityCurve") -> None:
        """
        Add a fragility curve for a specific moment_survived value.

        Used during building phase before saving to disk.

        Args:
            moment_survived: Moment threshold for this curve.
            curve: FragilityCurve instance.
        """
        self._curves[moment_survived] = curve

        # Update moment_survived_values list
        if self.moment_survived_values is None:
            self.moment_survived_values = np.array([moment_survived])
        elif moment_survived not in self.moment_survived_values:
            self.moment_survived_values = np.sort(np.append(self.moment_survived_values, moment_survived))

        # Set corrosion_ratios from first curve
        if self.corrosion_ratios is None:
            self.corrosion_ratios = curve.corrosion_ratios

    def _get_curve(self, moment_survived: float) -> "FragilityCurve":
        """
        Get fragility curve for exact moment_survived value.

        Loads from disk if not in cache.

        Args:
            moment_survived: Exact moment threshold (must be in grid).

        Returns:
            FragilityCurve instance.
        """
        # Check in-memory curves first (building phase)
        if moment_survived in self._curves:
            return self._curves[moment_survived]

        # Check cache
        if moment_survived in self._cache:
            return self._cache[moment_survived]

        # Load from disk
        if self.directory is None:
            raise ValueError("No directory set and curve not in memory.")

        filepath = self.directory / f"moment_{moment_survived:.1f}.json"
        if not filepath.exists():
            raise FileNotFoundError(f"Fragility curve not found: {filepath}")

        curve = FragilityCurve.load(filepath)
        self._cache[moment_survived] = curve
        return curve

    def _find_bracket(self, moment_survived: float) -> Tuple[float, float, float]:
        """
        Find bracketing moment values and interpolation weight.

        Args:
            moment_survived: Target moment threshold.

        Returns:
            Tuple of (moment_lo, moment_hi, t) where t is interpolation weight.
            If exact match, returns (moment, moment, 0.0).
        """
        if self.moment_survived_values is None or len(self.moment_survived_values) == 0:
            raise ValueError("No moment_survived values available.")

        moments = self.moment_survived_values

        # Clamp to range
        if moment_survived <= moments[0]:
            return moments[0], moments[0], 0.0
        if moment_survived >= moments[-1]:
            return moments[-1], moments[-1], 0.0

        # Find bracketing indices
        idx = np.searchsorted(moments, moment_survived)
        moment_lo = moments[idx - 1]
        moment_hi = moments[idx]

        # Interpolation weight
        t = (moment_survived - moment_lo) / (moment_hi - moment_lo)

        return moment_lo, moment_hi, t

    def pf_at(self, corrosion_ratio: float, moment_survived: float) -> float:
        """
        Interpolate Pf at given (corrosion_ratio, moment_survived).

        Loads only the necessary curve(s) from disk.

        Args:
            corrosion_ratio: Corrosion ratio [0, 1].
            moment_survived: Proven moment capacity threshold [kNm].

        Returns:
            Interpolated failure probability.
        """
        moment_lo, moment_hi, t = self._find_bracket(moment_survived)

        if t == 0.0:
            # Exact match or clamped to boundary
            curve = self._get_curve(moment_lo)
            return curve.pf_at(corrosion_ratio)

        # Interpolate between two curves
        curve_lo = self._get_curve(moment_lo)
        curve_hi = self._get_curve(moment_hi)

        pf_lo = curve_lo.pf_at(corrosion_ratio)
        pf_hi = curve_hi.pf_at(corrosion_ratio)

        return (1 - t) * pf_lo + t * pf_hi

    def beta_at(self, corrosion_ratio: float, moment_survived: float) -> float:
        """
        Interpolate reliability index at given (corrosion_ratio, moment_survived).

        Args:
            corrosion_ratio: Corrosion ratio [0, 1].
            moment_survived: Proven moment capacity threshold [kNm].

        Returns:
            Interpolated reliability index.
        """
        pf = self.pf_at(corrosion_ratio, moment_survived)
        pf_clipped = np.clip(pf, 1e-10, 1 - 1e-10)
        return float(-stats.norm.ppf(pf_clipped))

    def get_curve_at(self, moment_survived: float) -> "FragilityCurve":
        """
        Get interpolated fragility curve at given moment_survived.

        If exact match in grid, returns that curve. Otherwise, creates
        a new curve by interpolating between adjacent curves.

        Args:
            moment_survived: Proven moment capacity threshold [kNm].

        Returns:
            FragilityCurve instance (may be interpolated).
        """
        moment_lo, moment_hi, t = self._find_bracket(moment_survived)

        if t == 0.0:
            return self._get_curve(moment_lo)

        # Interpolate to create new curve
        curve_lo = self._get_curve(moment_lo)
        curve_hi = self._get_curve(moment_hi)

        pf_interp = (1 - t) * curve_lo.pf + t * curve_hi.pf

        return FragilityCurve(
            corrosion_ratios=self.corrosion_ratios.copy(),
            pf=pf_interp,
            metadata={
                "interpolated": True,
                "moment_survived": moment_survived,
                "moment_lo": moment_lo,
                "moment_hi": moment_hi,
                "t": t,
            },
        )

    def integrate_pf(
        self,
        corrosion_ratio_pdf: NDArray,
        moment_survived: float,
    ) -> float:
        """
        Compute total Pf by integrating over corrosion ratio distribution.

        Args:
            corrosion_ratio_pdf: PDF values at self.corrosion_ratios grid.
            moment_survived: Proven moment capacity threshold [kNm].

        Returns:
            Total failure probability.
        """
        curve = self.get_curve_at(moment_survived)
        return curve.integrate_pf(corrosion_ratio_pdf)

    def clear_cache(self) -> None:
        """Clear the loaded curves cache to free memory."""
        self._cache.clear()

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def save(self, directory: Path | str) -> None:
        """
        Save all curves to directory as JSON files.

        Args:
            directory: Output directory for fragility curves.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save manifest
        manifest = {
            "corrosion_ratios": self.corrosion_ratios.tolist(),
            "moment_survived_values": self.moment_survived_values.tolist(),
            "metadata": self.metadata,
            "created_at": datetime.now().isoformat(),
        }
        with open(directory / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        # Save each curve as JSON
        for moment_val, curve in self._curves.items():
            filepath = directory / f"moment_{moment_val:.1f}.json"
            curve.save(filepath, fmt="json")

        self.directory = directory

    @classmethod
    def load(cls, directory: Path | str) -> "FragilitySurfaceIndex":
        """
        Load index from directory (does not load curves until needed).

        Args:
            directory: Directory containing manifest.json and curve files.

        Returns:
            FragilitySurfaceIndex instance.
        """
        directory = Path(directory)

        # Load manifest
        with open(directory / "manifest.json", "r") as f:
            manifest = json.load(f)

        return cls(
            directory=directory,
            corrosion_ratios=np.array(manifest["corrosion_ratios"]),
            moment_survived_values=np.array(manifest["moment_survived_values"]),
            metadata=manifest.get("metadata", {}),
        )

    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        n_cr = len(self.corrosion_ratios) if self.corrosion_ratios is not None else 0
        n_moments = len(self.moment_survived_values) if self.moment_survived_values is not None else 0
        cached = len(self._cache) + len(self._curves)
        return (
            f"FragilitySurfaceIndex("
            f"n_cr={n_cr}, n_moments={n_moments}, cached={cached})"
        )

    def summary(self) -> str:
        """Return a text summary of the fragility surface index."""
        lines = ["FragilitySurfaceIndex Summary"]

        if self.corrosion_ratios is not None:
            lines.append(f"  Corrosion ratio grid: {len(self.corrosion_ratios)} points")
            lines.append(
                f"    Range: [{self.corrosion_ratios.min():.3f}, "
                f"{self.corrosion_ratios.max():.3f}]"
            )

        if self.moment_survived_values is not None:
            lines.append(
                f"  Moment survived grid: {len(self.moment_survived_values)} points"
            )
            lines.append(
                f"    Range: [{self.moment_survived_values.min():.1f}, "
                f"{self.moment_survived_values.max():.1f}] kNm"
            )

        lines.append(f"  Curves in memory: {len(self._curves)}")
        lines.append(f"  Curves cached: {len(self._cache)}")

        if self.directory:
            lines.append(f"  Directory: {self.directory}")

        return "\n".join(lines)


@dataclass
class FragilityCurve:
    """
    Fragility curve: Pf as function of corrosion ratio.

    Attributes:
        corrosion_ratios: Grid of corrosion ratio values.
        pf: Failure probabilities at each corrosion ratio.
        beta: Reliability indices at each corrosion ratio.
        max_moments: Cached max moment samples per corrosion ratio (optional).
        metadata: Additional metadata (creation time, parameters, etc.).
    """

    corrosion_ratios: NDArray
    pf: NDArray
    beta: NDArray = field(init=False)
    max_moments: Optional[NDArray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Compute reliability index from Pf."""
        # Clip Pf to avoid inf values
        pf_clipped = np.clip(self.pf, 1e-10, 1 - 1e-10)
        self.beta = -stats.norm.ppf(pf_clipped)

        # Add creation timestamp if not present
        if "created_at" not in self.metadata:
            self.metadata["created_at"] = datetime.now().isoformat()

    def pf_at(self, corrosion_ratio: float) -> float:
        """Interpolate Pf at given corrosion ratio."""
        return float(np.interp(corrosion_ratio, self.corrosion_ratios, self.pf))

    def beta_at(self, corrosion_ratio: float) -> float:
        """Interpolate beta at given corrosion ratio."""
        return float(np.interp(corrosion_ratio, self.corrosion_ratios, self.beta))

    def integrate_pf(self, corrosion_ratio_pdf: NDArray) -> float:
        """
        Compute total Pf by integrating over corrosion ratio distribution.

        Args:
            corrosion_ratio_pdf: PDF values at self.corrosion_ratios grid.

        Returns:
            Total failure probability.
        """
        return float(np.trapezoid(self.pf * corrosion_ratio_pdf, self.corrosion_ratios))

    def recompute_pf(
        self,
        moment_cap: float,
        weights: Optional[NDArray] = None,
    ) -> NDArray:
        """
        Recompute Pf from cached max_moments with custom weights.

        Use this after applying proven strength or other sample reweighting.

        Args:
            moment_cap: Initial moment capacity [kNm].
            weights: Sample weights (n_samples,). If None, uses uniform weights.

        Returns:
            Array of Pf values at each corrosion ratio.
        """
        if self.max_moments is None:
            raise ValueError("max_moments not cached. Rebuild with cache_moments=True.")

        n_samples = self.max_moments.shape[1]
        if weights is None:
            weights = np.ones(n_samples)

        pf = np.zeros(len(self.corrosion_ratios))
        for i, cr in enumerate(self.corrosion_ratios):
            moment_cap_degraded = moment_cap * (1 - cr)
            g = moment_cap_degraded - self.max_moments[i]
            pf[i] = np.sum((g < 0) * weights) / np.sum(weights)

        return pf

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self, include_moments: bool = False) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        Args:
            include_moments: Include max_moments array (can be large).

        Returns:
            Dictionary representation.
        """
        data = {
            "corrosion_ratios": self.corrosion_ratios.tolist(),
            "pf": self.pf.tolist(),
            "beta": self.beta.tolist(),
            "metadata": self.metadata,
        }
        if include_moments and self.max_moments is not None:
            data["max_moments"] = self.max_moments.tolist()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FragilityCurve":
        """
        Create FragilityCurve from dictionary.

        Args:
            data: Dictionary with corrosion_ratios, pf, and optional fields.

        Returns:
            FragilityCurve instance.
        """
        max_moments = data.get("max_moments")
        if max_moments is not None:
            max_moments = np.array(max_moments)

        return cls(
            corrosion_ratios=np.array(data["corrosion_ratios"]),
            pf=np.array(data["pf"]),
            max_moments=max_moments,
            metadata=data.get("metadata", {}),
        )

    def to_json(self, include_moments: bool = False) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(include_moments=include_moments), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "FragilityCurve":
        """Create FragilityCurve from JSON string."""
        return cls.from_dict(json.loads(json_str))

    # -------------------------------------------------------------------------
    # File I/O
    # -------------------------------------------------------------------------

    def save(self, filepath: Path | str, fmt: str = "npz") -> None:
        """
        Save fragility curve to file.

        Args:
            filepath: Output file path.
            fmt: Format - "npz" (binary, includes moments) or "json" (portable).
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "json":
            with open(filepath, "w") as f:
                json.dump(self.to_dict(include_moments=False), f, indent=2)

        elif fmt == "npz":
            data = {
                "corrosion_ratios": self.corrosion_ratios,
                "pf": self.pf,
                "beta": self.beta,
                "metadata": np.array([json.dumps(self.metadata)]),
            }
            if self.max_moments is not None:
                data["max_moments"] = self.max_moments
            np.savez(filepath, **data)

        else:
            raise ValueError(f"Unknown format: {fmt}. Use 'npz' or 'json'.")

    @classmethod
    def load(cls, filepath: Path | str) -> "FragilityCurve":
        """
        Load fragility curve from file.

        Automatically detects format from file extension.

        Args:
            filepath: Input file path (.npz or .json).

        Returns:
            FragilityCurve instance.
        """
        filepath = Path(filepath)

        if filepath.suffix == ".json":
            with open(filepath, "r") as f:
                return cls.from_dict(json.load(f))

        elif filepath.suffix in (".npz", ".npy"):
            data = np.load(filepath, allow_pickle=True)

            # Extract metadata
            metadata = {}
            if "metadata" in data:
                metadata = json.loads(str(data["metadata"][0]))

            # Extract max_moments if present
            max_moments = None
            if "max_moments" in data:
                max_moments = data["max_moments"]

            return cls(
                corrosion_ratios=data["corrosion_ratios"],
                pf=data["pf"],
                max_moments=max_moments,
                metadata=metadata,
            )

        else:
            raise ValueError(f"Unknown file extension: {filepath.suffix}")

    def save_summary(self, filepath: Path | str) -> None:
        """
        Save a lightweight summary (no max_moments) to JSON.

        Args:
            filepath: Output JSON file path.
        """
        self.save(filepath, fmt="json")

    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"FragilityCurve("
            f"n_points={len(self.corrosion_ratios)}, "
            f"cr=[{self.corrosion_ratios.min():.3f}, {self.corrosion_ratios.max():.3f}], "
            f"pf=[{self.pf.min():.2e}, {self.pf.max():.2e}])"
        )

    def summary(self) -> str:
        """Return a text summary of the fragility curve."""
        lines = [
            f"FragilityCurve Summary",
            f"  Grid points: {len(self.corrosion_ratios)}",
            f"  CR range: [{self.corrosion_ratios.min():.3f}, {self.corrosion_ratios.max():.3f}]",
            f"  Pf range: [{self.pf.min():.2e}, {self.pf.max():.2e}]",
            f"  Beta range: [{self.beta.min():.2f}, {self.beta.max():.2f}]",
            f"  Max moments cached: {self.max_moments is not None}",
        ]
        if self.metadata:
            lines.append(f"  Metadata: {list(self.metadata.keys())}")
        return "\n".join(lines)


class MLP(nn.Module):
    """
    Multi-Layer Perceptron surrogate model.

    Architecture: [Linear -> ReLU]* + [Linear -> Tanh]
    """

    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Performance(BasePerformance):
    """
    Performance function for sheet pile wall with corrosion.

    The limit state function is:
        g = moment_capacity * (1 - corrosion_ratio(t)) - max_moment(x, EI_degraded)

    Uses a pre-computed FragilityCurve for fast evaluation. The fragility curve
    caches max_moment predictions at each corrosion_ratio grid point, avoiding
    repeated surrogate calls.

    Args:
        name: Name of the performance function.
        parameters: Dictionary containing:
            - moment_cap: Initial moment capacity [kNm].
            - start_thickness: Initial wall thickness [mm].
            - corrosion_rate: Corrosion rate [mm/year] for time-based degradation.
            - ei_column_idx: Index of EI column in input samples.
    """

    def __init__(self, name: str, parameters: dict = {}):
        super().__init__(name, parameters)
        self.surrogate = None
        self.fragility: Optional[FragilityCurve] = None
        self._device = torch.device("cpu")

    def set_surrogate(self, model: nn.Module, scaler_x: Any, scaler_y: Any) -> None:
        """
        Set the surrogate model after initialization.

        Args:
            model: Trained PyTorch model.
            scaler_x: Fitted input scaler.
            scaler_y: Fitted output scaler.
        """
        self.model = model
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.surrogate = (model, scaler_x, scaler_y)
        self.model.eval()
        self._device = next(self.model.parameters()).device

    def set_fragility(self, fragility: FragilityCurve) -> None:
        """
        Set the fragility curve for fast evaluation.

        The fragility curve must have max_moments cached. Once set, _lsf()
        will interpolate from cached values instead of calling the surrogate.

        Args:
            fragility: Pre-computed fragility curve with cached max_moments.
        """
        if fragility.max_moments is None:
            raise ValueError("FragilityCurve must have max_moments cached.")
        self.fragility = fragility

    def predict_moment(self, x: NDArray) -> NDArray:
        """
        Predict max moment using surrogate model.

        Args:
            x: Input samples (n_samples, n_features).

        Returns:
            Predicted max moments (n_samples,).
        """
        if self.surrogate is None:
            raise ValueError("Surrogate model not set. Use set_surrogate() first.")

        x_scaled = self.scaler_x.transform(x)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=self._device)

        with torch.no_grad():
            y_scaled = self.model(x_tensor)

        y_scaled_np = y_scaled.cpu().numpy()
        y = self.scaler_y.inverse_transform(y_scaled_np)

        return y.squeeze()

    def _lsf(self, x: NDArray, t: float = 0) -> Tuple[NDArray, Dict[str, Any]]:
        """
        Evaluate the limit state function.

        If a fragility curve is set, uses cached max_moments (fast).
        Otherwise, calls the surrogate model directly.

        Args:
            x: Input samples (n_samples, n_features).
                Expected columns: soil params, EI, water_lvl.
            t: Time [years]. Used to compute corrosion-based degradation.

        Returns:
            Tuple of (g, metadata) where:
                - g: Performance function values (n_samples,). g < 0 means failure.
                - metadata: Dict with intermediate values (empty for consistency).
        """
        x_ = np.atleast_2d(x)

        # Get parameters
        moment_cap = self.parameters.get("moment_cap", 750.0)
        start_thickness = self.parameters.get("start_thickness", 10.0)
        corrosion_rate = self.parameters.get("corrosion_rate", 0.1)

        # Compute corrosion ratio from time
        thickness_loss = corrosion_rate * t
        corrosion_ratio = min(thickness_loss / start_thickness, 1.0)

        # Degraded capacity
        moment_cap_degraded = moment_cap * (1 - corrosion_ratio)

        # Get max_moment: from fragility curve or surrogate
        if self.fragility is not None:
            # Interpolate from cached max_moments
            max_moment = self._interpolate_moments(corrosion_ratio)
        else:
            # Direct evaluation
            max_moment = self._compute_max_moment(x_, corrosion_ratio)

        # Limit state: g < 0 means failure
        g = moment_cap_degraded - max_moment

        return g, {}

    def _interpolate_moments(self, corrosion_ratio: float) -> NDArray:
        """
        Interpolate max_moments from fragility curve at given corrosion ratio.

        Args:
            corrosion_ratio: Corrosion ratio [0, 1].

        Returns:
            Interpolated max_moment for each sample.
        """
        cr_grid = self.fragility.corrosion_ratios
        moments = self.fragility.max_moments  # (n_grid, n_samples)

        # Find bracketing indices
        idx = np.searchsorted(cr_grid, corrosion_ratio)
        if idx == 0:
            return moments[0]
        if idx >= len(cr_grid):
            return moments[-1]

        # Linear interpolation
        cr_lo, cr_hi = cr_grid[idx - 1], cr_grid[idx]
        t_interp = (corrosion_ratio - cr_lo) / (cr_hi - cr_lo)
        return (1 - t_interp) * moments[idx - 1] + t_interp * moments[idx]

    def _compute_max_moment(self, x: NDArray, corrosion_ratio: float) -> NDArray:
        """
        Compute max_moment directly (via surrogate or fallback).

        Args:
            x: Input samples (n_samples, n_features).
            corrosion_ratio: Corrosion ratio [0, 1].

        Returns:
            Max moment for each sample.
        """
        ei_column_idx = self.parameters.get("ei_column_idx", -2)

        # Degrade stiffness
        x_degraded = x.copy()
        EI_original = x_degraded[:, ei_column_idx]
        EI_degraded = EI_original * (1 - corrosion_ratio)
        x_degraded[:, ei_column_idx] = EI_degraded

        if self.surrogate is not None:
            return self.predict_moment(x_degraded)
        elif hasattr(self, '_max_moments_cache') and self._max_moments_cache is not None:
            # Use cached ground truth max_moments (for testing with real data)
            return self._max_moments_cache
        else:
            raise ValueError("No means of estimating the generated moments.")

    def set_max_moments_cache(self, max_moments: NDArray) -> None:
        """Set ground truth max_moments for testing (bypasses surrogate)."""
        self._max_moments_cache = max_moments

    def _grad(self, x: NDArray, t: float = 0) -> NDArray:
        """Gradient not implemented for surrogate-based model."""
        raise NotImplementedError("Gradients not available for surrogate model.")

    def lsf_at_corrosion_ratio(
        self,
        x: NDArray,
        corrosion_ratio: float,
    ) -> Tuple[NDArray, NDArray]:
        """
        Evaluate LSF at a specific corrosion ratio (for building fragility curves).

        This bypasses the time-based corrosion calculation in _lsf.

        Args:
            x: Input samples (n_samples, n_features).
            corrosion_ratio: Direct corrosion ratio [0, 1].

        Returns:
            Tuple of (g, max_moment) arrays.
        """
        x_ = np.atleast_2d(x)
        moment_cap = self.parameters.get("moment_cap", 750.0)
        moment_cap_degraded = moment_cap * (1 - corrosion_ratio)

        # Compute max_moment
        max_moment = self._compute_max_moment(x_, corrosion_ratio)

        g = moment_cap_degraded / max_moment - 1
        return g, max_moment

    def failure_probability(
        self,
        x: NDArray,
        corrosion_ratio: float,
        weights: Optional[NDArray] = None,
    ) -> float:
        """
        Estimate failure probability via Monte Carlo.

        Args:
            x: Input samples (n_samples, n_features).
            corrosion_ratio: Corrosion ratio for this evaluation.
            weights: Sample weights for importance sampling.

        Returns:
            Estimated failure probability.
        """
        g, _ = self.lsf_at_corrosion_ratio(x, corrosion_ratio)

        if weights is None:
            weights = np.ones(len(g))

        pf = np.sum((g < 0) * weights) / np.sum(weights)
        return pf

    def build_fragility_curve(
        self,
        x: NDArray,
        corrosion_ratios: Optional[NDArray] = None,
        n_grid: int = 100,
        weights: Optional[NDArray] = None,
        cache_moments: bool = True,
        verbose: bool = False,
    ) -> FragilityCurve:
        """
        Build fragility curve: Pf as function of corrosion ratio.

        Args:
            x: Input samples (n_samples, n_features).
            corrosion_ratios: Grid of corrosion ratios. If None, uses linspace(0, 1, n_grid).
            n_grid: Number of grid points if corrosion_ratios not provided.
            weights: Sample weights for importance sampling.
            cache_moments: Store max_moments per corrosion ratio (memory intensive).
            verbose: Print progress.

        Returns:
            FragilityCurve with Pf and beta at each corrosion ratio.
        """
        if corrosion_ratios is None:
            corrosion_ratios = np.linspace(0, 1, n_grid)

        x = np.atleast_2d(x)
        n_samples = x.shape[0]
        n_ratios = len(corrosion_ratios)

        if weights is None:
            weights = np.ones(n_samples)

        pf = np.zeros(n_ratios)
        max_moments = np.zeros((n_ratios, n_samples)) if cache_moments else None

        for i, cr in enumerate(corrosion_ratios):
            if verbose and i % 10 == 0:
                print(f"Computing Pf for corrosion_ratio {i+1}/{n_ratios}: {cr:.3f}")

            g, max_moment = self.lsf_at_corrosion_ratio(x, cr)
            pf[i] = np.sum((g < 0) * weights) / np.sum(weights)

            if cache_moments:
                max_moments[i] = max_moment

        # Build metadata
        metadata = {
            "performance_name": self.name,
            "n_samples": n_samples,
            "n_grid": n_ratios,
            "cache_moments": cache_moments,
            "parameters": self.parameters.copy(),
        }

        return FragilityCurve(
            corrosion_ratios=corrosion_ratios,
            pf=pf,
            max_moments=max_moments,
            metadata=metadata,
        )

    def build_fragility_surface(
        self,
        x: NDArray,
        corrosion_ratios: Optional[NDArray] = None,
        moment_survived_values: Optional[NDArray] = None,
        n_cr: int = 100,
        n_moments: int = 50,
        moment_range: Optional[Tuple[float, float]] = None,
        verbose: bool = False,
    ) -> FragilitySurfaceIndex:
        """
        Build 2D fragility surface as collection of 1D curves.

        Pre-computes a FragilityCurve for each moment_survived threshold.
        Each curve stores Pf vs corrosion_ratio conditioned on survival.
        Curves are stored separately for memory-efficient on-demand loading.

        The proven strength logic: samples whose max_moment exceeds the
        moment_survived threshold are excluded (they would have failed before).
        Pf is computed among the remaining "surviving" samples.

        Args:
            x: Input samples (n_samples, n_features).
            corrosion_ratios: Grid of corrosion ratios. If None, uses linspace(0, 1, n_cr).
            moment_survived_values: Grid of moment thresholds. If None, auto-computed.
            n_cr: Number of corrosion ratio grid points.
            n_moments: Number of moment threshold grid points.
            moment_range: (min, max) for moment_survived grid. If None, auto-computed
                from sample moment range with 10% padding.
            verbose: Print progress.

        Returns:
            FragilitySurfaceIndex with one FragilityCurve per moment_survived.
        """
        x = np.atleast_2d(x)
        n_samples = x.shape[0]
        moment_cap = self.parameters.get("moment_cap", 750.0)

        # Set up corrosion ratio grid
        if corrosion_ratios is None:
            corrosion_ratios = np.linspace(0, 1, n_cr)
        n_cr = len(corrosion_ratios)

        # Compute max_moments at each corrosion ratio (this is the expensive part)
        if verbose:
            print(f"Computing max_moments at {n_cr} corrosion ratios...")

        max_moments_grid = np.zeros((n_cr, n_samples))
        for i, cr in enumerate(corrosion_ratios):
            if verbose and i % 10 == 0:
                print(f"  CR {i+1}/{n_cr}: {cr:.3f}")
            _, max_moment = self.lsf_at_corrosion_ratio(x, cr)
            max_moments_grid[i] = max_moment

        # Set up moment_survived grid
        if moment_survived_values is None:
            if moment_range is None:
                # Auto-compute from sample range with padding
                moment_min = max_moments_grid.min() * 0.9
                moment_max = max_moments_grid.max() * 1.1
                moment_range = (moment_min, moment_max)
                moment_survived_values = np.linspace(moment_range[0], moment_range[1], n_moments)


        if verbose:
            print(f"Building {n_moments} fragility curves...")

        # Create index
        index = FragilitySurfaceIndex(
            corrosion_ratios=corrosion_ratios,
            moment_survived_values=moment_survived_values,
            metadata={
                "performance_name": self.name,
                "n_samples": n_samples,
                "n_cr": n_cr,
                "n_moments": n_moments,
                "moment_range": list(moment_range) if moment_range else None,
                "parameters": self.parameters.copy(),
            },
        )

        # Build a FragilityCurve for each moment_survived
        for j, moment_survived in enumerate(moment_survived_values):
            if verbose and j % 10 == 0:
                print(f"  Moment {j+1}/{n_moments}: {moment_survived:.1f} kNm")

            # Compute Pf at each CR conditioned on survival
            pf = np.zeros(n_cr)
            for i, cr in enumerate(corrosion_ratios):
                max_moments = max_moments_grid[i]
                moment_cap_degraded = moment_cap * (1 - cr)

                survived = max_moments <= moment_survived
                n_survived = survived.sum()
                failed = max_moments > moment_cap_degraded

                if n_survived > 0:
                    pf[i] = np.dot(failed, survived) / n_survived
                else:
                    pf[i] = np.mean(failed)

            # Create curve and add to index
            curve = FragilityCurve(
                corrosion_ratios=corrosion_ratios.copy(),
                pf=pf,
                metadata={
                    "moment_survived": moment_survived,
                    "n_samples": n_samples,
                },
            )
            index.add_curve(moment_survived, curve)

        if verbose:
            print("FragilitySurfaceIndex built successfully.")
            print(index.summary())

        return index

    def pf_from_fragility(
        self,
        fragility: FragilityCurve,
        corrosion_ratio_pdf: NDArray,
    ) -> Tuple[float, float]:
        """
        Compute Pf by integrating fragility curve over corrosion ratio distribution.

        This avoids re-running MCS when only the corrosion distribution changes
        (e.g., after Bayesian updating).

        Args:
            fragility: Pre-computed fragility curve.
            corrosion_ratio_pdf: PDF of corrosion ratio at fragility.corrosion_ratios.

        Returns:
            Tuple of (pf, beta).
        """
        pf = fragility.integrate_pf(corrosion_ratio_pdf)
        pf_clipped = np.clip(pf, 1e-10, 1 - 1e-10)
        beta = -stats.norm.ppf(pf_clipped)
        return pf, beta

    def proven_strength_weights(
        self,
        t_survived: float,
        fragility: Optional[FragilityCurve] = None,
    ) -> NDArray:
        """
        Compute sample weights for proven strength (survival) conditioning.

        If the structure survived until t_survived, samples that would have
        failed by that time are impossible and get weight 0.

        This implements lower truncation of capacity: samples where
        max_moment > moment_cap_degraded at any t <= t_survived are excluded.

        Args:
            t_survived: Time the structure has survived [years].
            fragility: FragilityCurve with cached max_moments. Uses self.fragility if None.

        Returns:
            Weights array (n_samples,). 1.0 for surviving samples, 0.0 for failed.
        """
        if fragility is None:
            fragility = self.fragility
        if fragility is None or fragility.max_moments is None:
            raise ValueError("Fragility curve with cached max_moments required.")

        moment_cap = self.parameters.get("moment_cap", 750.0)
        start_thickness = self.parameters.get("start_thickness", 10.0)
        corrosion_rate = self.parameters.get("corrosion_rate", 0.1)

        # Corrosion ratio at survival time
        cr_survived = min((corrosion_rate * t_survived) / start_thickness, 1.0)

        # Find all corrosion ratios up to cr_survived
        cr_grid = fragility.corrosion_ratios
        mask_survived = cr_grid <= cr_survived

        # For each sample, check if it would have failed at any cr <= cr_survived
        n_samples = fragility.max_moments.shape[1]
        weights = np.ones(n_samples)

        for i, cr in enumerate(cr_grid[mask_survived]):
            moment_cap_degraded = moment_cap * (1 - cr)
            max_moments = fragility.max_moments[i]
            # Sample failed if max_moment > capacity
            failed = max_moments > moment_cap_degraded
            weights[failed] = 0.0

        return weights

    def proven_strength_weights_at_cr(
        self,
        cr_survived: float,
        fragility: Optional[FragilityCurve] = None,
    ) -> NDArray:
        """
        Compute sample weights for proven strength at a specific corrosion ratio.

        Simpler version: only checks survival at cr_survived, not the full history.

        Args:
            cr_survived: Corrosion ratio the structure has survived.
            fragility: FragilityCurve with cached max_moments.

        Returns:
            Weights array (n_samples,). 1.0 for surviving samples, 0.0 for failed.
        """
        if fragility is None:
            fragility = self.fragility
        if fragility is None or fragility.max_moments is None:
            raise ValueError("Fragility curve with cached max_moments required.")

        moment_cap = self.parameters.get("moment_cap", 750.0)
        moment_cap_degraded = moment_cap * (1 - cr_survived)

        # Interpolate max_moments at cr_survived (for each sample)
        cr_grid = fragility.corrosion_ratios
        moments_grid = fragility.max_moments  # (n_grid, n_samples)
        n_samples = moments_grid.shape[1]

        # Linear interpolation per sample
        idx = np.searchsorted(cr_grid, cr_survived)
        if idx == 0:
            max_moments = moments_grid[0]
        elif idx >= len(cr_grid):
            max_moments = moments_grid[-1]
        else:
            cr_lo, cr_hi = cr_grid[idx - 1], cr_grid[idx]
            t_interp = (cr_survived - cr_lo) / (cr_hi - cr_lo)
            max_moments = (1 - t_interp) * moments_grid[idx - 1] + t_interp * moments_grid[idx]

        # Samples that would have failed get weight 0
        weights = (max_moments <= moment_cap_degraded).astype(float)
        return weights

    def pf_with_proven_strength(
        self,
        t_survived: float,
        fragility: Optional[FragilityCurve] = None,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Compute Pf curve conditioned on proven strength (survival).

        Combines proven_strength_weights with fragility.recompute_pf.

        Args:
            t_survived: Time the structure has survived [years].
            fragility: FragilityCurve with cached max_moments.

        Returns:
            Tuple of (pf, beta, weights) where:
                - pf: Failure probabilities at each corrosion ratio.
                - beta: Reliability indices at each corrosion ratio.
                - weights: Sample weights (for further analysis).
        """
        if fragility is None:
            fragility = self.fragility
        if fragility is None:
            raise ValueError("Fragility curve required.")

        moment_cap = self.parameters.get("moment_cap", 750.0)

        # Get proven strength weights
        weights = self.proven_strength_weights(t_survived, fragility)

        # Recompute Pf with these weights
        pf = fragility.recompute_pf(moment_cap, weights)

        # Compute beta
        pf_clipped = np.clip(pf, 1e-10, 1 - 1e-10)
        beta = -stats.norm.ppf(pf_clipped)

        return pf, beta, weights


if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path
    from case_studies.dsheet_example import io

    # Load real surrogate training data
    surrogate_data = io.load_surrogate_data()

    # Input columns (11 features)
    input_cols = [
        "Klei_soilcohesion", "Klei_soilphi", "Klei_soilcurkb1",
        "Zand_soilphi", "Zand_soilcurkb1",
        "Zandvast_soilphi", "Zandvast_soilcurkb1",
        "Zandlos_soilphi", "Zandlos_soilcurkb1",
        "Wall_SheetPilingElementEI", "water_lvl",
    ]
    x = surrogate_data[input_cols].values
    n_samples = len(x)
    print(f"Loaded {n_samples} samples from surrogate_data.csv")

    # Moment columns for computing max_moment (ground truth)
    moment_cols = [c for c in surrogate_data.columns if c.startswith("moment_")]
    moments_all = surrogate_data[moment_cols].values
    max_moments_true = np.abs(moments_all).max(axis=1)
    print(f"  Max moment range: [{max_moments_true.min():.1f}, {max_moments_true.max():.1f}]")

    # Parameters from case study
    params = {
        "moment_cap": 750.0,
        "start_thickness": 9.5,
        "corrosion_rate": 0.022,
        "EI_start": 48800.92,  # From case_study.json true_params
        "ei_column_idx": 9,    # Wall_SheetPilingElementEI
    }

    perf = Performance(name="dsheet_test", parameters=params)

    # Load trained surrogate model
    print("\nLoading trained surrogate model...")
    model_kwargs = {
        "input_dim": 11,
        "hidden_dims": [1024, 512, 256, 128, 64, 32],
        "output_dim": 1,
    }
    model, scaler_x, scaler_y = io.load_surrogate_model(MLP, model_kwargs)
    perf.set_surrogate(model, scaler_x, scaler_y)
    print("  Surrogate model loaded and set.")

    # Step 1: Build fragility curve (expensive, done once)
    print("Building fragility curve (one-time cost)...")
    fragility = perf.build_fragility_curve(x, n_grid=50, cache_moments=True)
    print(f"  Built: {fragility}")

    # Step 2: Set fragility for fast evaluation
    perf.set_fragility(fragility)

    # Step 3: Evaluate at multiple timesteps (fast, uses cached moments)
    # Note: corrosion_rate=0.022 mm/year is slow; extend to 200 years to see Pf rise
    print("\nEvaluating g(x, t) for t=0..200 (uses fragility curve)...")
    g_t = {}
    for t in range(201):
        g_t[t], _ = perf.lsf(x, t=t)

    # Compute Pf over time
    pf_t = [np.mean(g < 0) for g in g_t.values()]
    beta_t = [-stats.norm.ppf(max(pf, 1e-10)) for pf in pf_t]

    print(f"  Pf at t=0:   {pf_t[0]:.4f} (beta={beta_t[0]:.2f})")
    print(f"  Pf at t=50:  {pf_t[50]:.4f} (beta={beta_t[50]:.2f})")
    print(f"  Pf at t=100: {pf_t[100]:.4f} (beta={beta_t[100]:.2f})")
    print(f"  Pf at t=150: {pf_t[150]:.4f} (beta={beta_t[150]:.2f})")
    print(f"  Pf at t=200: {pf_t[200]:.4f} (beta={beta_t[200]:.2f})")

    # Fragility curve utilities
    print(f"\nFragility curve interpolation:")
    print(f"  Pf at CR=0.3: {fragility.pf_at(0.3):.4f}")
    print(f"  Beta at CR=0.3: {fragility.beta_at(0.3):.2f}")

    # Integrate over a corrosion ratio distribution (e.g., after Bayesian update)
    cr_pdf = stats.norm.pdf(fragility.corrosion_ratios, loc=0.25, scale=0.1)
    cr_pdf /= np.trapezoid(cr_pdf, fragility.corrosion_ratios)
    pf_total, beta_total = perf.pf_from_fragility(fragility, cr_pdf)
    print(f"\n  Integrated Pf (posterior): {pf_total:.4f}")
    print(f"  Integrated Beta (posterior): {beta_total:.2f}")

    # Proven strength: structure survived until t_survived
    # Use t=150 where Pf is ~7% to show the filtering effect
    print("\n--- Proven Strength (survival conditioning) ---")
    t_survived = 150

    # Method 1: Get weights and recompute fragility curve Pf
    pf_proven_fc, beta_proven_fc, w_proven = perf.pf_with_proven_strength(t_survived, fragility)
    n_surviving = int(np.sum(w_proven > 0))
    print(f"  Structure survived t={t_survived} years")
    print(f"  Samples consistent with survival: {n_surviving}/{len(w_proven)} ({100*n_surviving/len(w_proven):.1f}%)")

    # Method 2: Or use fragility.recompute_pf directly
    pf_recomputed = fragility.recompute_pf(params["moment_cap"], w_proven)

    # Compare fragility curves
    print(f"\n  Fragility curve Pf (prior vs proven strength):")
    print(f"    CR=0.3: prior={fragility.pf_at(0.3):.4f}, proven={np.interp(0.3, fragility.corrosion_ratios, pf_proven_fc):.4f}")
    print(f"    CR=0.5: prior={fragility.pf_at(0.5):.4f}, proven={np.interp(0.5, fragility.corrosion_ratios, pf_proven_fc):.4f}")

    # Use weights for time-based Pf (from g_t)
    pf_t_proven = [np.sum((g < 0) * w_proven) / np.sum(w_proven) for g in g_t.values()]
    beta_t_proven = [-stats.norm.ppf(max(pf, 1e-10)) for pf in pf_t_proven]

    print(f"\n  Time-based Pf (prior vs proven strength at t={t_survived}):")
    for t in [50, 80, 100, 150, 200]:
        print(f"    t={t}: prior={pf_t[t]:.4f} (β={beta_t[t]:.2f}), proven={pf_t_proven[t]:.4f} (β={beta_t_proven[t]:.2f})")

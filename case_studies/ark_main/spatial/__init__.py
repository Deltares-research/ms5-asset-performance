"""Spatial-variability MCS for a sheet-pile wall.

Output layout per run, under ``<remote>/output/spatial_analysis/``::

    cached/<setup_signature>/        # pf_grid.json, posterior_grid.json
    results/<setup_signature>/       # summary.json, forecasts/, plots/

The signature encodes the run setup (LSF, n_samples, seed, geometry, basic-
variable kernel, cr kernel, and a hash over per-variable overrides if any),
so different configs produce different folders and identical re-runs land
right back on the same one.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def setup_signature(spatial_settings: dict) -> str:
    """Build a folder-name signature from a spatial_settings dict.

    The signature is a single underscore-joined string that picks out
    everything a downstream consumer would call a "config":

    * ``lsf-<name>`` — selected fragility cache.
    * ``N<n_samples>``, ``seed<seed>`` — MC budget and RNG seed.
    * ``L<int>``, ``n<int>`` — wall length and section count.
    * ``wth<int>``, ``wrh<g>`` — wall (basic-variable) kernel theta, rho_0.
    * ``crth<int>``, ``crrh<g>`` — cr-field kernel theta, rho_0.
    * ``pv-<8-hex>`` — short hash of the ``per_variable`` overrides dict
      (only appended if it's non-empty).

    Floats use ``:g`` so trailing zeros are dropped (``0.30 → "0.3"``);
    integer-valued floats are cast with ``int()`` to keep the names short
    (``200.0 → "200"``).
    """
    s = spatial_settings
    # Strip the conventional ``lsf_`` prefix so the signature reads
    # ``lsf-wall_...`` instead of ``lsf-lsf_wall_...``.
    lsf_tag = s["lsf_name"]
    if lsf_tag.startswith("lsf_"):
        lsf_tag = lsf_tag[len("lsf_"):]
    parts = [
        f"lsf-{lsf_tag}",
        f"N{int(s['n_samples'])}",
        f"seed{int(s['seed'])}",
        f"L{int(s['L'])}",
        f"n{int(s['n_sections'])}",
        f"wth{int(s['wall']['theta'])}",
        f"wrh{s['wall']['rho_0']:g}",
        f"crth{int(s['cr']['theta'])}",
        f"crrh{s['cr']['rho_0']:g}",
    ]
    pv = s.get("per_variable") or {}
    if pv:
        h = hashlib.sha1(
            json.dumps(pv, sort_keys=True).encode()
        ).hexdigest()[:8]
        parts.append(f"pv-{h}")
    return "_".join(parts)


def cache_dir(remote: Path, spatial_settings: dict) -> Path:
    """``<remote>/output/spatial_analysis/cached/<signature>/``."""
    return (
        remote / "output" / "spatial_analysis"
        / "cached" / setup_signature(spatial_settings)
    )


def results_dir(remote: Path, spatial_settings: dict) -> Path:
    """``<remote>/output/spatial_analysis/results/<signature>/``."""
    return (
        remote / "output" / "spatial_analysis"
        / "results" / setup_signature(spatial_settings)
    )

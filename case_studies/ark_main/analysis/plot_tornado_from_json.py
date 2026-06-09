"""
Render a tornado plot for one LSF component directly from an existing
``sensitivity_<lsf>/cr_<rate>.json`` file (produced by run_sensitivity.py).

This is a pure post-processing step: it does NOT re-run D-SheetPiling, it
just reads the cached g(p01)/g(p99) values and draws the tornado. Useful for
regenerating / restyling the plot without the solver in the loop.

Reuses the same bar layout as run_sensitivity.plot_tornado (bars span
g_p01..g_p99, sorted by |dg|, coloured by which side moves on p99, baseline
and g=0 reference lines) and adds human-readable variable labels with units.

Usage::

    python analysis/plot_tornado_from_json.py \
        --json "P:/.../sensitivity_lsf_wall_anchor/cr_0.0000.json" \
        --component anchor
"""
from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# Human-readable labels (mirrors io/export_wall_params).
_SOIL_PROPERTY_LABEL = {
    "phi":      ("φ",     "°"),
    "cohesion": ("c",     "kPa"),
    "gamdry":   ("γ_dry", "kN/m³"),
    "gamwet":   ("γ_wet", "kN/m³"),
    "curkb1":   ("k_b1",  "kN/m³"),
}
_OTHER_LABEL = {
    "model_factor_M":            ("θ_M",            "-"),
    "model_factor_F":            ("θ_F",            "-"),
    "phreatic_level":            ("Phreatic level", "mNAP"),
    "canal_level":               ("Canal level",    "mNAP"),
    "uniform_load_left":         ("Uniform load",   "kN/m"),
    "Wall_SheetPilingElementEI": ("Wall EI",        "kNm²/m"),
}


def pretty_variable(name: str) -> str:
    if name in _OTHER_LABEL:
        sym, unit = _OTHER_LABEL[name]
        return f"{sym} ({unit})"
    if "_soil" in name:
        layer, _, prop = name.partition("_soil")
        if prop in _SOIL_PROPERTY_LABEL:
            sym, unit = _SOIL_PROPERTY_LABEL[prop]
            return f"{layer} {sym} ({unit})"
    return name.replace("_", " ")


def load_stochastic_vars(settings_path: Path) -> set[str]:
    """Names kept stochastic in the analysis settings file.

    A variable is stochastic unless its distribution_type is
    deterministic/constant/fixed or its standard_deviation is 0. This is the
    same rule the FORM/MCS builders use, so the returned set is exactly the
    variables that vary in the reliability analysis (settings.json), as
    opposed to the un-pruned settings_full.json.
    """
    s = json.load(open(settings_path))
    stoch: set[str] = set()
    for v in s.get("variables", []):
        dt = v.get("distribution_type", "normal").lower()
        det = dt in ("deterministic", "constant", "fixed") \
            or float(v.get("standard_deviation", 0.0)) == 0.0
        if not det:
            stoch.add(v["name"])
    return stoch


def plot_tornado_from_json(json_path: Path, component: str,
                           out_path: Path | None = None,
                           stochastic_vars: set[str] | None = None) -> Path:
    data = json.load(open(json_path))
    comps = data["components"]
    if component not in comps:
        raise ValueError(f"Component {component!r} not in {comps}")
    c_idx = comps.index(component)
    g_baseline_c = float(data["g_baseline"][c_idx])

    rows = [
        (r["variable"],
         pretty_variable(r["variable"]),
         float(r["g_p01"][c_idx]),
         float(r["g_p99"][c_idx]),
         float(r["dg"][c_idx]))
        for r in data["variables"]
    ]
    # Drop variables with zero effect on this component (e.g. model_factor_M
    # has dg = 0 for the anchor — it only acts on the wall).
    rows = [r for r in rows if abs(r[4]) > 1e-12]
    rows.sort(key=lambda x: abs(x[4]), reverse=True)

    raw = [r[0] for r in rows]
    names = [r[1] for r in rows]
    g_lo = np.array([min(r[2], r[3]) for r in rows])
    g_hi = np.array([max(r[2], r[3]) for r in rows])
    width = g_hi - g_lo
    sign = np.array([np.sign(r[4]) for r in rows])
    absdg = np.abs(np.array([r[4] for r in rows]))

    # Stochastic (kept) vs fixed-at-mean per the analysis settings.
    if stochastic_vars is None:
        is_stoch = np.ones(len(rows), dtype=bool)
        have_screen = False
    else:
        is_stoch = np.array([nm in stochastic_vars for nm in raw])
        have_screen = True

    n = len(rows)
    fig_h = max(3.5, 0.34 * n + 1.8)
    fig, ax = plt.subplots(figsize=(9.5, fig_h))

    y = np.arange(n)
    # Stochastic bars coloured by sign; fixed bars greyed out.
    colors = []
    for s, keep in zip(sign, is_stoch):
        if not keep:
            colors.append("#c9c9c9")
        else:
            colors.append("#4c8dde" if s >= 0 else "#d6604d")
    ax.barh(y, width, left=g_lo, color=colors, alpha=0.9,
            edgecolor="k", linewidth=0.5)

    # Annotate each bar with its dg magnitude.
    for yi, (lo, hi) in enumerate(zip(g_lo, g_hi)):
        ax.text(hi + 0.01 * (g_hi.max() - g_lo.min() + 1e-9), yi,
                f"Δg={hi - lo:.2f}", va="center", ha="left", fontsize=7.5,
                color="#333" if is_stoch[yi] else "#999")

    ax.axvline(g_baseline_c, color="k", linestyle="--", linewidth=1.2)
    xlim_lo, xlim_hi = ax.get_xlim()
    show_zero = xlim_lo <= 0.0 <= xlim_hi
    if show_zero:
        ax.axvline(0.0, color="#a00", linestyle=":", linewidth=1.4)

    # --- screening margin: where stochastic (kept) meets fixed-at-mean ---
    # Rows are sorted by |dg| descending; with a clean importance cut the
    # kept variables sit on top. Draw a horizontal separator between the last
    # kept and the first fixed row, and shade the "fixed" band.
    margin_handle = None
    if have_screen and is_stoch.any() and (~is_stoch).any():
        kept_rows = np.where(is_stoch)[0]
        fixed_rows = np.where(~is_stoch)[0]
        contiguous = kept_rows.max() < fixed_rows.min()
        if contiguous:
            y_cut = kept_rows.max() + 0.5
            # midpoint |dg| of the cut, for the label
            dg_cut = 0.5 * (absdg[kept_rows.max()] + absdg[fixed_rows.min()])
            ax.axhline(y_cut, color="#2a7a2a", linestyle="-", linewidth=1.6)
            ax.axhspan(y_cut, n - 0.5, color="#9e9e9e", alpha=0.12, zorder=0)
            ax.text(0.015, 0.02,
                    f"below line: fixed at mean (|Δg| ≲ {dg_cut:.2f})",
                    transform=ax.transAxes, fontsize=8, color="#2a7a2a",
                    va="bottom", ha="left")
            margin_handle = Line2D([], [], color="#2a7a2a", linewidth=1.6,
                                   label="stochastic / fixed screening cut")

    # Explicit legend handles so the two bar colours render correctly.
    handles = [
        Patch(facecolor="#4c8dde", edgecolor="k", linewidth=0.5,
              label="stochastic: g rises with variable"),
        Patch(facecolor="#d6604d", edgecolor="k", linewidth=0.5,
              label="stochastic: g falls with variable"),
    ]
    if have_screen:
        handles.append(Patch(facecolor="#c9c9c9", edgecolor="k", linewidth=0.5,
                             label="fixed at mean (excluded)"))
    handles.append(Line2D([], [], color="k", linestyle="--", linewidth=1.2,
                          label=f"baseline g = {g_baseline_c:.2f}"))
    if show_zero:
        handles.append(Line2D([], [], color="#a00", linestyle=":",
                              linewidth=1.4, label="g = 0 (failure)"))
    if margin_handle is not None:
        handles.append(margin_handle)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    # Grey + italicise the fixed-variable tick labels.
    for lbl, keep in zip(ax.get_yticklabels(), is_stoch):
        if not keep:
            lbl.set_color("#999")
            lbl.set_style("italic")
    ax.invert_yaxis()
    ax.set_xlabel(f"g_{component}")
    cr = data.get("corrosion_rate", 0.0)
    ax.set_title(
        f"Tornado — {data['lsf_name']} / {component} (cr={cr:.2f})\n"
        f"one-at-a-time 1%↔99%, sorted by |Δg|"
        + ("   (colour = stochastic in analysis; grey = fixed at mean)"
           if have_screen else ""),
        fontsize=10.5,
    )
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    if out_path is None:
        out_path = (json_path.parent / "plots" /
                    f"tornado_{component}_cr_{cr:.4f}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--json", type=Path, required=True,
                   help="Path to sensitivity cr_<rate>.json")
    p.add_argument("--component", type=str, default="anchor",
                   help="LSF component to plot (e.g. wall, anchor)")
    p.add_argument("--out", type=Path, default=None,
                   help="Output PNG (default: <json dir>/plots/tornado_<c>_cr_<rate>.png)")
    p.add_argument("--analysis-settings", type=Path, default=None,
                   help="Path to the analysis settings.json (the PRUNED, "
                        "stochastic set). When given, variables fixed at their "
                        "mean in that file are greyed out and a screening "
                        "separator is drawn. Omit to colour all bars.")
    args = p.parse_args()
    stoch = (load_stochastic_vars(args.analysis_settings)
             if args.analysis_settings else None)
    out = plot_tornado_from_json(args.json, args.component, args.out,
                                 stochastic_vars=stoch)
    print(f"Wrote {out}")

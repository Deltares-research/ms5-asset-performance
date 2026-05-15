"""
Step 6 of the spatial analysis — integrate the cr-parametric Pf grid over a
distribution of cr to get the Pf at one time t.

The Pf grid comes in on the fragility cr-grid (sparse). The cr-pdf typically
lives on a finer grid (from ``run.py``'s corrosion-model output). Linear
interpolation of ``Pf(cr)`` onto the pdf grid + trapezoidal integration is
the standard recipe and matches what the per-section pipeline already does.

The t = 0 case is special-cased: ``cr_pdf(t = 0) = delta(cr = 0)`` collapses
the integral to a point evaluation of the Pf grid at cr = 0, which avoids
having to materialise a numerical Dirac delta.
"""
from __future__ import annotations

import numpy as np


def over_cr(
    pf_section_grid: np.ndarray,        # (n_cr, n_sections)
    pf_system_grid: np.ndarray,         # (n_cr,)
    cr_values: np.ndarray,              # (n_cr,) — fragility-cache cr grid
    *,
    cr_pdf_grid: np.ndarray | None = None,
    cr_pdf_values: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Integrate spatial Pf grid over a cr distribution.

    Returns ``(pf_section_t, pf_system_t)`` — per-section and system Pf at
    the integrated time t.

    If neither ``cr_pdf_grid`` nor ``cr_pdf_values`` is given, treats the
    cr distribution as ``delta(cr = 0)`` (the t = 0 case) and returns the
    Pf values at the fragility point closest to cr = 0.
    """
    if cr_pdf_grid is None or cr_pdf_values is None:
        idx0 = int(np.argmin(np.abs(cr_values - 0.0)))
        return pf_section_grid[idx0, :].copy(), float(pf_system_grid[idx0])

    n_sec = pf_section_grid.shape[1]
    pf_sec_fine = np.column_stack([
        np.interp(cr_pdf_grid, cr_values, pf_section_grid[:, i])
        for i in range(n_sec)
    ])  # (n_pdf, n_sections)
    pf_sys_fine = np.interp(cr_pdf_grid, cr_values, pf_system_grid)

    pf_section_t = np.trapezoid(
        pf_sec_fine * cr_pdf_values[:, None], cr_pdf_grid, axis=0,
    )
    pf_system_t = float(np.trapezoid(pf_sys_fine * cr_pdf_values, cr_pdf_grid))
    return pf_section_t, pf_system_t

"""Diagnostic: evaluate lsf_wall at mean inputs across cr.

If g(mean) is positive at cr=0..0.4 and crosses zero around cr=0.5, the LSF
math is correct and any negative beta at low cr is purely a FORM convergence
bug, not an LSF bug.

Usage (PyCharm console, with cwd = case_studies/ark_main):
    runfile('analysis/check_lsf.py')
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from reliability.build_fragility import (
    _smooth_g_wall, init_model, lsf_wall, _settings, _config,
    SMOOTH_ALPHA,
)

init_model(use_api=False)

variables = _settings["variables"]
mean_kw = {v["name"]: float(v["mean"]) for v in variables}

print(f"Smooth-alpha = {SMOOTH_ALPHA}")
print(f"{'cr':>5} {'g_smooth(mean)':>16} {'safe?':>6}")
for cr in np.linspace(0.0, 0.8, 9):
    kw = dict(mean_kw)
    kw["corrosion_rate"] = float(cr)
    g = lsf_wall(**kw)
    print(f"{cr:>5.2f} {g:>+16.4f} {'YES' if g > 0 else 'no':>6}")

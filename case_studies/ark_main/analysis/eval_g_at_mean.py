"""Run lsf_wall at all-mean inputs across cr to find where g_wall(mean) = 0."""
import json
from copy import deepcopy
from pathlib import Path
from dotenv import load_dotenv
import os
import numpy as np

from src.io import get_remote_path
from src.geotechnical_models.dsheetpiling.model import DSheetPiling
from src.reliability_models.dsheetpiling import build_payload, apply_payload

_ENV = Path(__file__).resolve().parents[1] / ".env"
remote = get_remote_path(_ENV)
with open(remote / "input" / "settings.json") as f:
    s = json.load(f)
config = s["parameters"]
variables = s["variables"]

# Build mean-input kwargs
def mean_kwargs():
    kw = {v["name"]: float(v["mean"]) for v in variables}
    kw.pop("model_factor_M", None)
    kw.pop("model_factor_F", None)
    return kw

base = DSheetPiling(str(remote / "input" / "model.shi"), api_key=None)
M_cap = float(config["wall_moment_capacity"])
F_yield = float(config["anchor_capacity"])
EI_start = float(config["EI_start"])

print(f'{"cr":>5} {"|M|_max":>9} {"M_cap*(1-cr)":>13} {"g_wall(mean,thM=1)":>20}'
      f'   {"|F|":>7} {"F_yield":>8} {"g_anchor(mean,thF=1)":>20}')
for cr in np.linspace(0.0, 0.8, 9):
    factor = 1.0 - cr
    kw = mean_kwargs()
    kw["Wall_SheetPilingElementEI"] = EI_start * factor
    model = deepcopy(base)
    payload = build_payload(kw, model)
    apply_payload(model, payload)
    model.execute()
    moment = model.results.moment[0]
    if isinstance(moment, np.ndarray):
        moment = moment.tolist()
    M_max = max(abs(m) for m in moment)
    F = model.results.anchor_force[0]
    if isinstance(F, (list, np.ndarray)):
        F = float(np.asarray(F).item())
    F_abs = abs(F)
    cap = M_cap * factor
    g_w = cap / M_max - 1
    g_a = F_yield / F_abs - 1
    print(f'{cr:>5.2f} {M_max:>9.1f} {cap:>13.1f} {g_w:>+20.4f}'
          f'   {F_abs:>7.1f} {F_yield:>8.1f} {g_a:>+20.4f}')

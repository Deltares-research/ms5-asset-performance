import json
base = r"P:\ms5-smart-and-resilient-infra\asset-performance\Case studies 2026\ARK main\output"

print(f'{"idx":>3} {"cr":>5}  '
      f'{"|M|_w":>7} {"|M|_wa":>7} {"d|M|":>6}   '
      f'{"|F|_w":>6} {"|F|_wa":>6} {"d|F|":>5}   '
      f'{"theta_M_w":>9} {"theta_M_wa":>10}')
for i in range(9):
    with open(rf"{base}\fragility_curve_lsf_wall\internals\point_{i:04d}.json") as f:
        w = json.load(f)
    with open(rf"{base}\fragility_curve_lsf_wall_anchor\internals\point_{i:04d}.json") as f:
        a = json.load(f)
    with open(rf"{base}\fragility_curve_lsf_wall\point_{i:04d}.json") as f:
        wp = json.load(f)
    with open(rf"{base}\fragility_curve_lsf_wall_anchor\point_{i:04d}.json") as f:
        ap = json.load(f)
    cr = w["cr"]
    mw, ma = abs(w["max_moment"]), abs(a["max_moment"])
    fw, fa = abs(w["anchor_force"]), abs(a["anchor_force"])
    tw = wp["design_point"].get("model_factor_M", 1.0)
    ta = ap["design_point"].get("model_factor_M", 1.0)
    print(f'{i:>3} {cr:>5.2f}  '
          f'{mw:>7.1f} {ma:>7.1f} {ma-mw:>+6.1f}   '
          f'{fw:>6.1f} {fa:>6.1f} {fa-fw:>+5.1f}   '
          f'{tw:>9.4f} {ta:>10.4f}')

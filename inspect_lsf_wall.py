import json
base = r"P:\ms5-smart-and-resilient-infra\asset-performance\Case studies 2026\ARK main\output\fragility_curve_lsf_wall"

print(f'{"i":>2} {"cr":>5} {"beta":>7} {"method":>14} {"converge":>9}  '
      f'{"Klei_phi":>9} {"Zand_phi":>9} {"Zandvast_phi":>13}  '
      f'{"theta_M":>8} {"phr_lvl":>8} {"canal":>7} {"q":>6}')
for i in range(11):
    try:
        with open(rf"{base}\point_{i:04d}.json") as f:
            p = json.load(f)
    except FileNotFoundError:
        continue
    dp = p.get("design_point", {})
    print(f'{i:>2} {p["point"]["corrosion_rate"]:>5.2f} {p["beta"]:>7.3f} {p["method"]:>14} {str(p["convergence"]):>9}  '
          f'{dp.get("Klei_soilphi", 0):>9.3f} {dp.get("Zand_soilphi", 0):>9.3f} {dp.get("Zandvast_soilphi", 0):>13.3f}  '
          f'{dp.get("model_factor_M", 0):>8.4f} {dp.get("phreatic_level", 0):>8.3f} {dp.get("canal_level", 0):>7.3f} {dp.get("uniform_load_left", 0):>6.2f}')

# Also check manifest
print()
print("Manifest:")
with open(rf"{base}\manifest.json") as f:
    m = json.load(f)
print("  form_params:", json.dumps(m.get("form_params", {}), indent=2))

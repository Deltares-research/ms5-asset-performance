import json
base = r"P:\ms5-smart-and-resilient-infra\asset-performance\Case studies 2026\ARK main\output"

def load(name, idx):
    with open(rf"{base}\fragility_curve_{name}\point_{idx:04d}.json") as f:
        return json.load(f)

print(f'{"idx":>3} {"cr":>5} {"beta_wall":>10} {"beta_wa":>10} {"pf_wall":>11} {"pf_wa":>11} {"ratio":>7}')
for i in range(11):
    try:
        w = load("lsf_wall", i)
        a = load("lsf_wall_anchor", i)
    except FileNotFoundError:
        continue
    cr = w["point"]["corrosion_rate"]
    print(f'{i:>3} {cr:>5.2f} {w["beta"]:>10.4f} {a["beta"]:>10.4f} {w["pf"]:>11.3e} {a["pf"]:>11.3e} {a["pf"]/w["pf"]:>7.2f}')

print()
w = load("lsf_wall", 0)
a = load("lsf_wall_anchor", 0)
print(f'{"variable":<28} {"wall":>12} {"wa":>12} {"diff":>10}')
for k in w["design_point"]:
    wv = w["design_point"][k]
    av = a["design_point"].get(k, 0)
    if abs(wv) + abs(av) > 0:
        print(f"{k:<28} {wv:>12.4f} {av:>12.4f} {av-wv:>+10.4f}")

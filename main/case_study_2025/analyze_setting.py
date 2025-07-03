import json
from pathlib import Path

from mpmath import monitor

if __name__ == "__main__":

    path = Path("data/setting")

    with open(path/"case_study.json", "r") as f:
        data = json.load(f)

    zs = [val["z"] for val in data.values()]
    same_zs = all(z == zs[0] for z in zs)
    if same_zs:
        print("All z's are the same!")
        with open(path/"z.json", "w") as f:
            json.dump(zs[0], f)
    else:
        print("Not all z's are the same")

    monitoring_zs = [val["z_monitoring"] for val in data.values()]
    same_monitoring_zs = all(z == monitoring_zs[0] for z in monitoring_zs)
    if same_zs:
        print("All monitoring z's are the same!")
        with open(path/"z_monitoring.json", "w") as f:
            json.dump(monitoring_zs[0], f)
    else:
        print("Not all monitoring z's are the same")


import os
from src.geotechnical_models.dsheetpiling.model import DSheetPiling


if __name__ == "__main__":

    model_path = os.environ["MODEL_PATH"]  # model_path defined as environment variable
    result_path = r"results/results.json"
    soil_data = {"Klei": {"soilcohesion": 10.}}
    water_data = {"A": +1.}
    load_data = {"load": (15, 0.)}

    model = DSheetPiling(model_path)
    model.update_soils(soil_data)
    model.update_water(water_data)
    model.update_uniform_loads(load_data)
    model.execute(result_path)

    benchmark_model = DSheetPiling(model_path)
    benchmark_model.load_results(r"results/bench_results.json")
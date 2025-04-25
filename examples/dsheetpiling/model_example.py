import os
from src.geotechnical_models.dsheetpiling.model import DSheetPiling


if __name__ == "__main__":

    geomodel_path = os.environ["MODEL_PATH"]  # model_path defined as environment variable
    result_path = r"results/results.json"
    soil_data = {"Klei": {"soilcohesion": 10.}}
    water_data = {"A": +1.}
    load_data = {"load": (15, 0.)}

    geomodel = DSheetPiling(geomodel_path)
    geomodel.update_soils(soil_data)
    geomodel.update_water(water_data)
    geomodel.update_uniform_loads(load_data)
    geomodel.execute(result_path)

    benchmark_model = DSheetPiling(geomodel_path)
    benchmark_model.load_results(r"results/bench_results.json")
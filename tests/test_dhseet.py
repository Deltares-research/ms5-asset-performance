import os
from src.geotechnical_models.dsheetpiling.model import DSheetPiling

def test_execution():

    model_path = os.environ["MODEL_PATH"]  # model_path defined as environment variable
    result_path = r"../examples/dsheet_model/results.json"
    soil_data = {"Klei": {"soilcohesion": 10.}}
    water_data = {"GWS  0,0": +1.}
    load_data = {"A": (15, 0.)}

    model = DSheetPiling(model_path)
    model.update_soils(soil_data)
    model.update_water(water_data)
    model.update_uniform_loads(load_data)
    model.execute(result_path)

    benchmark_model = DSheetPiling(model_path)
    benchmark_model.load_results(r"../examples/dsheet_model/bench_results.json")

    assert model.results.__eq__(benchmark_model.results)


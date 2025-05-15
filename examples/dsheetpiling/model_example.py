import os
from src.geotechnical_models.dsheetpiling.model import DSheetPiling
from src.corrosion.corrosion_model import CorrosionModel


if __name__ == "__main__":

    geomodel_path = os.environ["MODEL_PATH"]  # model_path defined as environment variable
    result_path = r"results/results.json"
    soil_data = {"Klei": {"soilcohesion": 10.}}
    water_data = {"A": +1.}
    load_data = {"load": (15, 0.)}
    wall_data = {"SheetPilingElementEI": 9999.}
    anchor_data = {"Emod": 9999.}

    C50 = 1.5
    time = 65.
    corrosion_model = CorrosionModel(corrosion_rate=0.022, start_thickness=9.5)
    corrosion = corrosion_model.generate_observations(time, C50).squeeze()

    geomodel = DSheetPiling(geomodel_path)
    geomodel.update_soils(soil_data)
    geomodel.update_water(water_data)
    geomodel.update_uniform_loads(load_data)
    geomodel.update_wall(wall_data)
    geomodel.apply_corrosion(corrosion, corrosion_model.start_thickness)
    geomodel.update_anchor(anchor_data)
    geomodel.execute(result_path)

    benchmark_model = DSheetPiling(geomodel_path)
    benchmark_model.load_results(r"results/bench_results.json")


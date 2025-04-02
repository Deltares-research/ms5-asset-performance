import os
from utils import DSheetPilingResults, DSheetPilingStageResults, WaterData, WaterLevel
from copy import deepcopy
from src.geotechnical_models.base import GeoModelBase
from geolib.models.dsheetpiling import DSheetPilingModel
from geolib.models.dsheetpiling.internal import SoilCollection
from pathlib import Path
from typing import Optional, Dict, List, Tuple


class DSheetPiling(GeoModelBase):

    def __init__(self, model_path: str | Path, exe_path: Optional[str | Path] = None) -> None:
        super(GeoModelBase, self).__init__()
        if not isinstance(model_path, Path): model_path = Path(model_path)
        self.model_path = Path(model_path.as_posix())
        if not self.model_path.exists():
            raise NotADirectoryError("Model path does not exist.")
        if exe_path is not None:
            if not isinstance(exe_path, Path): exe_path = Path(exe_path)
        else:
            exe_path = model_path.with_name(model_path.stem+"_executed"+model_path.suffix)
        self.exe_path = Path(exe_path.as_posix())
        if not exe_path.parent.exists(): os.mkdir(self.exe_path.parent)
        self.parse_model(self.model_path)

    def parse_model(self, path: str | Path) -> None:
        if not isinstance(path, Path): path = Path(path)
        geomodel = DSheetPilingModel()
        geomodel.parse(path)
        self.geomodel = geomodel
        self.n_stages = int(self.geomodel.input.input_data.construction_stages[0].split(" ")[0])
        self.soils = self.get_soils()
        self.wall = self.get_wall()
        self.water = self.get_water()

    def get_soils(self) -> Dict[str, SoilCollection]:
        return {soil.name: soil for soil in deepcopy(self.geomodel.input.input_data.soil_collection.soil)}

    def update_soils(self, soil_data: Dict[str, Dict[str, float]]) -> None:
        for (soil_name, soil_params) in soil_data.items():
            for (soil_param_name, soil_param_value) in soil_params.items():
                if hasattr(self.soils[soil_name], soil_param_name):
                    setattr(self.soils[soil_name], soil_param_name, float(soil_param_value))
                else:
                    raise AttributeError(f"Soil parameter {soil_param_name} not found in {soil_name}.")
        self.geomodel.input.input_data.soil_collection.soil = list(self.soils.values())

    def get_water(self) -> WaterData:
        water_input =  deepcopy(self.geomodel.input.input_data.waterlevels)
        return WaterData(water_input)
    
    def update_water(self, water_lvls: Dict[str, float]) -> None:
        self.water.adjust(water_lvls)
        water_lines = self.water.write()
        self.geomodel.input.input_data.waterlevels = water_lines

    def get_wall(self) -> float:
        pass

    def execute(self) -> None:
        self.geomodel.serialize(self.exe_path)  # _executed model is parsed from now on. TODO: Check w/ Eleni
        self.geomodel.execute()  # Make sure to add 'geolib.env' in run directory
        self.results = self.read_dsheet_results()

    def read_dsheet_results(self) -> DSheetPilingResults:

        stage_result_lst = []
        for i_stage, stage in enumerate(self.geomodel.output.construction_stage):
            stage_num = i_stage + 1
            results = stage.moments_forces_displacements.momentsforcesdisplacements
            wall_points = [list(point.values())[0]
                           for point in self.geomodel.output.points_on_sheetpile[i_stage].pointsonsheetpile]
            moments = [res['moment'] for res in results]
            shear_forces = [res['shear_force'] for res in results]
            displacements = [res['displacements'] for res in results]

            stage_result = DSheetPilingStageResults(
                stage_id=stage_num,
                z=wall_points,
                moment=moments,
                shear=shear_forces,
                displacement=displacements,
            )
            stage_result_lst.append(stage_result)

        # TODO: Read anchor results.

        if len(stage_result_lst) != self.n_stages:
            error_message = (f"Parsing results discovered {len(stage_result_lst)} stages,"
                             f" but D-SheetPiling model has {self.n_stages} stages.")
            raise ValueError(error_message)

        results = DSheetPilingResults()
        results.read(stage_result_lst)

        return results

    def save_results(self, path: str | Path) -> None:
        if not isinstance(path, Path): path = Path(path)
        path = Path(path.as_posix())
        self.results.save_json(path)

    def load_results(self, path: str | Path) -> None:
        if not isinstance(path, Path): path = Path(path)
        path = Path(path.as_posix())
        self.results = DSheetPilingResults()
        self.results.load_json(path)


if __name__ == "__main__":

    model_path = os.environ["MODEL_PATH"]  # model_path defined as environment variable
    result_path = r"../../../results/example_results.json"
    soil_data = {"Klei": {"soilcohesion": 10.}}
    water_lvls = {"GWS  0,0": +1.}

    model = DSheetPiling(model_path)
    model.update_soils(soil_data)
    model.update_water(water_lvls)
    model.execute()
    model.save_results(result_path)

    loaded_model = DSheetPiling(model_path)
    loaded_model.load_results(result_path)

    model_check = model.results.__eq__(loaded_model.results)
    if model_check:
        print(f"Were the results loaded correctly?  -->  {model_check} \u2705")
    else:
        print(f"Were the results loaded correctly?  -->  {model_check} \u274C")


import os
from utils import DSheetPilingResults, DSheetPilingStageResults, WaterData, WaterLevel
from copy import deepcopy
from src.geotechnical_models.base import GeoModelBase
from geolib.models.dsheetpiling import DSheetPilingModel
from geolib.models.dsheetpiling.internal import SoilCollection, UniformLoad
from pathlib import Path
from typing import Optional
import json


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
        self.water = self.get_water()
        self.uniform_loads = self.get_uniform_loads()

    def get_soils(self) -> dict[str, SoilCollection]:
        return {soil.name: soil for soil in deepcopy(self.geomodel.input.input_data.soil_collection.soil)}

    def update_soils(self, soil_data: dict[str, dict[str, float]]) -> None:
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
    
    def update_water(self, water_lvls: dict[str, float]) -> None:
        self.water.adjust(water_lvls)
        water_lines = self.water.write()
        self.geomodel.input.input_data.waterlevels = water_lines

    def get_wall(self) -> float:
        raise NotImplementedError("Adjusting the structural properties of the wall is not possible yet.")

    def update_wall(self) -> float:
        raise NotImplementedError("Adjusting the structural properties of the wall is not possible yet.")

    def get_uniform_loads(self) -> dict[str, UniformLoad]:
        return {
            uniform_load.name: uniform_load
            for uniform_load in deepcopy(self.geomodel.input.input_data.uniform_loads.loads)
        }

    def update_uniform_loads(self, load_data: dict[str, list[float] | tuple[float,...]]) -> None:
        for (load_name, load_params) in load_data.items():
            load_left, load_right = load_params
            if load_name in list(load_data.keys()):
                self.uniform_loads[load_name].uniformloadleft = float(load_left)
                self.uniform_loads[load_name].uniformloadright = float(load_right)
            else:
                raise AttributeError(f"Uniform load name {load_name} not found in Uniform load list.")
        self.geomodel.input.input_data.uniform_loads.loads = list(self.uniform_loads.values())

    def execute(self, result_path: Optional[str | Path] = None) -> None:
        self.geomodel.serialize(self.exe_path)  # _executed model is parsed from now on. TODO: Check w/ Eleni
        self.geomodel.execute()  # Make sure to add 'geolib.env' in run directory
        self.results = self.read_dsheet_results()
        if result_path is not None:
            if not isinstance(result_path, Path): result_path = Path(result_path)
            result_path = Path(result_path.as_posix())
            if not result_path.exists():
                raise NotADirectoryError("Result path does not exist.")
            self.save_results(result_path)
            log_path = result_path.parent / "log.json"
            self.log_input(log_path)

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

    def log_input(self, path: str | Path) -> None:
        soil_dict = {soil_name: soil.__dict__ for (soil_name, soil) in self.soils.items()}
        water_lvl_dict = {water_lvl.name: water_lvl.lvl for water_lvl in self.water.water_lvls}
        uniform_load_dict = {uniform_load_name: uniform_load.__dict__ for (uniform_load_name, uniform_load) in self.uniform_loads.items()}
        log = {
            "soil": soil_dict,
            "water_levels": water_lvl_dict,
            "uniform_loads": uniform_load_dict,
        }
        with open(path, 'w') as f:
            json.dump(log, f)


if __name__ == "__main__":

    model_path = os.environ["MODEL_PATH"]  # model_path defined as environment variable
    result_path = r"../../../results/example_results.json"
    soil_data = {"Klei": {"soilcohesion": 10.}}
    water_data = {"GWS  0,0": +1.}
    load_data = {"A": (15, 0.)}

    model = DSheetPiling(model_path)
    model.update_soils(soil_data)
    model.update_water(water_data)
    model.update_uniform_loads(load_data)
    model.execute(result_path)

    loaded_model = DSheetPiling(model_path)
    loaded_model.load_results(result_path)

    model_check = model.results.__eq__(loaded_model.results)
    if model_check:
        print(f"Were the results loaded correctly?  -->  {model_check} \u2705")
    else:
        print(f"Were the results loaded correctly?  -->  {model_check} \u274C")


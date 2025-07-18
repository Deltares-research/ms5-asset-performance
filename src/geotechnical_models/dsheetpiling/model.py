import os
from src.geotechnical_models.dsheetpiling.utils import *
from copy import deepcopy
from src.geotechnical_models.base import GeoModelBase
from geolib.models.dsheetpiling import DSheetPilingModel
from geolib.models.dsheetpiling.internal import SoilCollection, UniformLoad
from pathlib import Path
from typing import Optional, Dict
import json
import warnings
from datetime import datetime


class DSheetPiling(GeoModelBase):

    def __init__(self, model_path: str | Path, exe_path: Optional[str | Path] = None) -> None:
        super(GeoModelBase, self).__init__()
        self.parse_model_path(model_path)
        self.parse_exe_path(exe_path)
        self.parse_model(self.model_path)

    def parse_model_path(self, model_path: str | Path) -> None:
        if not isinstance(model_path, Path): model_path = Path(Path(model_path).as_posix())
        if not model_path.exists():
            raise NotADirectoryError("Model path does not exist.")
        self.model_path = Path(model_path.as_posix())
        self.file_suffix = self.model_path.suffix
        self.file_name = self.model_path.stem

    def parse_exe_path(self, exe_path: Optional[str | Path] = None) -> None:
        if exe_path is not None:
            if not isinstance(exe_path, Path): exe_path = Path(Path(exe_path).as_posix())
            if not exe_path.exists(): os.mkdir(exe_path)
        else:
            exe_path = Path(self.model_path).parent / "execution"
            if not exe_path.exists(): os.mkdir(exe_path)
        self.exe_path = Path(exe_path.as_posix())

    def parse_model(self, path: str | Path) -> None:
        if not isinstance(path, Path): path = Path(Path(path).as_posix())
        geomodel = DSheetPilingModel()
        geomodel.parse(path)
        self.geomodel = geomodel
        self.n_stages = int(self.geomodel.input.input_data.construction_stages[0].split(" ")[0])
        self.soils = self.get_soils()
        self.water = self.get_water()
        self.uniform_loads = self.get_uniform_loads()
        self.wall = self.get_wall()
        self.anchor = self.get_anchor()

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

    def get_wall(self) -> WallProperties:

        sheet_piling_data = self.geomodel.input.input_data.sheet_piling
        sheet_piling_lines = sheet_piling_data.splitlines()[6:-1]
        sheet_piling_props = {
            key: float(value) if value != "" else 0.
            for (key, value) in [line.split("=") for line in sheet_piling_lines]
        }
        sheet_piling_props = {
            key: int(value) if value.is_integer() else value
            for (key, value) in sheet_piling_props.items()
        }
        wall_props = WallProperties(**sheet_piling_props)
        return wall_props

    def update_wall(self, new_wall_props: Dict[str, float]) -> None:

        updated_wall_props = self.wall._asdict()
        for (key, new_val) in new_wall_props.items():
            updated_wall_props[key] = new_val
        self.wall = WallProperties(**updated_wall_props)

        sheet_piling_data = self.geomodel.input.input_data.sheet_piling
        sheet_piling_lines = sheet_piling_data.splitlines()

        wall_lines = [key+"="+str(value) for (key, value) in updated_wall_props.items()]
        wall_lines = sheet_piling_lines[:6] + wall_lines + [sheet_piling_lines[-1]]
        wall_data = "\n".join(wall_lines)

        self.geomodel.input.input_data.sheet_piling = wall_data

    def apply_corrosion(self, corrosion: float, start_thickness: float) -> None:

        if isinstance(corrosion, np.ndarray): corrosion = corrosion.item()
        if isinstance(corrosion, list) or isinstance(corrosion, tuple): corrosion = corrosion[0]

        reduction_factor = corrosion / start_thickness
        EI = self.wall.SheetPilingElementEI * (1 - reduction_factor)
        M = self.wall.SheetPilingElementResistingMoment * (1 - reduction_factor)
        width = self.wall.SheetPilingPileWidth - corrosion

        wall_dict = self.wall._asdict()
        wall_dict["SheetPilingElementEI"] = EI
        wall_dict["SheetPilingElementResistingMoment"] = M
        wall_dict["SheetPilingPileWidth"] = width

        self.update_wall(wall_dict)

    def get_anchor(self) -> AnchorProperties:
        anchor_data = self.geomodel.input.input_data.anchors
        anchor_lines = anchor_data.splitlines()
        anchor_props_names = anchor_lines[1].split()
        idx = anchor_props_names.index("Cross")
        anchor_props_names[idx] = "Cross_sect"
        anchor_props_names.remove("sect.")
        idx = anchor_props_names.index("E-mod")
        anchor_props_names[idx] = "Emod"
        anchor_props_values = []
        for value in anchor_lines[2].split():
            try:
                value = float(value)
                value = int(value) if value.is_integer() else value
            except:
                pass
            anchor_props_values.append(value)
        anchor_props = {key: value for (key, value) in zip(anchor_props_names, anchor_props_values)}
        anchor_props = AnchorProperties(**anchor_props)
        return anchor_props

    def update_anchor(self, new_anchor_props: Dict[str, float]) -> None:

        updated_anchor_props = self.anchor._asdict()
        for (key, new_val) in new_anchor_props.items():
            updated_anchor_props[key] = new_val
        self.anchor = AnchorProperties(**updated_anchor_props)

        anchor_data = self.geomodel.input.input_data.anchors
        anchor_lines = anchor_data.splitlines()

        anchor_props_names = anchor_lines[1].split()
        idx = anchor_props_names.index("Cross")
        anchor_props_names[idx] = "Cross_sect"
        anchor_props_names.remove("sect.")
        idx = anchor_props_names.index("E-mod")
        anchor_props_names[idx] = "Emod"

        updated_anchor_line = [str(updated_anchor_props[name]) for name in anchor_props_names]
        updated_anchor_line = " " + " ".join(updated_anchor_line)
        updated_anchor_lines = anchor_lines[:2] + [updated_anchor_line]
        updated_anchor_data = "\n".join(updated_anchor_lines)

        self.geomodel.input.input_data.anchors = updated_anchor_data

    def execute(self, result_path: Optional[str | Path] = None, i_run: Optional[int] = None) -> None:
        if i_run is None:
            now = datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            file_name = self.file_name + "_executed_" + timestamp + self.file_suffix
        else:
            file_name = self.file_name + f"_executed_run{i_run:d}" + self.file_suffix
        exe_path = self.exe_path / file_name
        geomodel = deepcopy(self.geomodel)
        geomodel.serialize(exe_path)  # _executed model is used from now on.
        geomodel.execute()  # Make sure to add 'geolib.env' in run directory
        self.results = self.read_dsheet_results(geomodel)

        if result_path is not None:
            if not isinstance(result_path, Path): result_path = Path(Path(result_path).as_posix())
            result_path = Path(result_path.as_posix())
            result_folder = result_path.parent
            if not result_folder.exists():
                raise NotADirectoryError("Result path does not exist.")
            self.save_results(result_path)
            log_path = result_path.parent / "log.json"
            self.log_input(log_path)

    def read_dsheet_results(self, geomodel: Optional[DSheetPilingModel] = None) -> DSheetPilingResults:

        if geomodel is None:
            geomodel = self.geomodel

        stage_result_lst = []
        for i_stage, stage in enumerate(geomodel.output.construction_stage):
            stage_num = i_stage + 1
            results = stage.moments_forces_displacements.momentsforcesdisplacements
            wall_points = [list(point.values())[0]
                           for point in geomodel.output.points_on_sheetpile[i_stage].pointsonsheetpile]
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

        # TODO: Read anchor examples.

        if len(stage_result_lst) != self.n_stages:
            warning_message = (f"Parsing examples discovered {len(stage_result_lst)} stages,"
                             f" but D-SheetPiling model has {self.n_stages} stages.")
            raise warnings.warn(warning_message, UserWarning)

        results = DSheetPilingResults()
        results.read(stage_result_lst)

        return results

    def save_results(self, path: str | Path) -> None:
        if not isinstance(path, Path): path = Path(Path(path).as_posix())
        path = Path(path.as_posix())
        self.results.save_json(path)

    def load_results(self, path: str | Path) -> None:
        if not isinstance(path, Path): path = Path(Path(path).as_posix())
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

    pass


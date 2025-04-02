import os
import numpy as np
import pandas as pd
import json
from copy import deepcopy
from geomodel import GeoModelBase
from geolib.models.dsheetpiling import DSheetPilingModel
from geolib.models.dsettlement.internal import SoilCollection
from dataclasses import dataclass, asdict, field
from pathlib import Path, WindowsPath
from numpy.typing import NDArray
from typing import List, Optional, Annotated, Dict, Tuple


@dataclass
class DSheetPilingStageResults:
    stage_id: int
    z: list | Annotated[NDArray[np.float64], ("n_points")]
    moment: list | Annotated[NDArray[np.float64], ("n_points")]
    shear: list | Annotated[NDArray[np.float64], ("n_points")]
    displacement: list | Annotated[NDArray[np.float64], ("n_points")]
    max_moment: float = field(init=False)
    max_shear: float = field(init=False)
    max_displacement: float = field(init=False)

    def __post_init__(self):
        self.max_moment = max([abs(moment) for moment in self.moment])
        self.max_shear = max([abs(shear) for shear in self.shear])
        self.max_displacement = max([abs(displacement) for displacement in self.displacement])


class DSheetPilingResults:

    z: Optional[List[float]] = None
    moment: Optional[List[List[float]]] = None
    shear: Optional[List[List[float]]] = None
    displacement: Optional[List[List[float]]] = None
    max_moment: List[float] = None
    max_shear: List[float] = None
    max_displacement: List[float] = None
    n_stages: int = None
    stage_results = None

    def __init__(self):
        pass

    def __eq__(self, other: "DSheetPilingResults") -> bool:
        return self.__dict__ == other.__dict__

    def read(self, stage_results: List[DSheetPilingStageResults]) -> None:
        self.stage_results = stage_results
        self.n_stages = len(stage_results)
        self.z = [stage_result.z for stage_result in stage_results][0]
        self.moment = [stage_result.moment for stage_result in stage_results]
        self.shear = [stage_result.shear for stage_result in stage_results]
        self.displacement = [stage_result.displacement for stage_result in stage_results]
        self.max_moment = [stage_result.max_moment for stage_result in stage_results]
        self.max_shear = [stage_result.max_shear for stage_result in stage_results]
        self.max_displacement = [stage_result.max_displacement for stage_result in stage_results]

    def to_dict(self) -> dict:
        stage_result_dicts = [asdict(stage_result) for stage_result in self.stage_results]
        keys = list(stage_result_dicts[0].keys())
        stage_result_dict = {key: [stage_result_dict[key] for stage_result_dict in stage_result_dicts] for key in keys}
        stage_result_dict["z"] = stage_result_dict["z"][0]
        return stage_result_dict

    def from_dict(self, result_dict) -> None:
        self.n_stages = len(list(result_dict.keys()))
        self.z = [res["z"] for res in result_dict.values()][0]
        self.moment = [res["moment"] for res in result_dict.values()]
        self.shear = [res["shear"] for res in result_dict.values()]
        self.displacements = [res["displacements"] for res in result_dict.values()]
        self.moment = [res["moment"] for res in result_dict.values()]
        self.max_moment = [res["max_moment"] for res in result_dict.values()]
        self.max_shear = [res["max_shear"] for res in result_dict.values()]
        self.max_displacement = [res["max_displacement"] for res in result_dict.values()]

    def save_json(self, path: str | WindowsPath | Path) -> None:
        if not isinstance(path, Path): path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f)

    def load_json(self, path: str | WindowsPath | Path) -> None:
        if not isinstance(path, Path): path = Path(path)
        with open(path, 'r') as f:
            stage_results = json.load(f)
        n_stages = len(stage_results["stage_id"])
        stage_results["z"] = [stage_results["z"] for _ in range(n_stages)]

        stage_result_lst = []

        for results in zip(*stage_results.values()):

            stage_id, z, moment, shear, displacement, max_moment, max_shear, max_displacement = results

            stage_result = DSheetPilingStageResults(
                stage_id=stage_id,
                z=z,
                moment=moment,
                shear=shear,
                displacement=displacement,
            )
            stage_result_lst.append(stage_result)

        self.read(stage_result_lst)


class DSheetPiling(GeoModelBase):

    def __init__(self, model_path: str | WindowsPath | Path, exe_path: Optional[str | WindowsPath | Path] = None
                 ) -> None:
        super(GeoModelBase, self).__init__()
        if not isinstance(model_path, Path): model_path = Path(model_path)
        self.model_path = Path(model_path.as_posix())
        if exe_path is not None:
            if not isinstance(exe_path, Path): exe_path = Path(exe_path)
        else:
            exe_path = model_path.with_name(model_path.stem+"_executed"+model_path.suffix)
        self.exe_path = Path(exe_path.as_posix())
        self.parse_model()

    def parse_model(self) -> None:
        geomodel = DSheetPilingModel()
        geomodel.parse(self.model_path)
        self.geomodel = geomodel
        self.n_stages = int(self.geomodel.input.input_data.construction_stages[0].split(" ")[0])
        self.soils = self.list_soils()

    def list_soils(self) -> Dict[str, SoilCollection]:
        return {soil.name: soil for soil in deepcopy(self.geomodel.input.input_data.soil_collection.soil)}

    def adjust_soil(self, soil_data: Dict[str, Dict[str, float]]) -> None:
        for (soil_name, soil_params) in soil_data.items():
            for (soil_param_name, soil_param_value) in soil_params.items():
                if hasattr(self.soils[soil_name], soil_param_name):
                    setattr(self.soils[soil_name], soil_param_name, soil_param_value)
                else:
                    raise AttributeError(f"Soil parameter {soil_param_name} not found in {soil_name}.")
        self.geomodel.input.input_data.soil_collection.soil = list(self.soils.values())

    def execute(self) -> None:
        self.geomodel.serialize(self.exe_path)
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

        results = DSheetPilingResults()
        results.read(stage_result_lst)

        return results

    def save_results(self, path: str | WindowsPath | Path) -> None:
        if not isinstance(path, Path): path = Path(path)
        path = Path(path.as_posix())
        self.results.save_json(path)

    def load_results(self, path: str | WindowsPath | Path) -> None:
        if not isinstance(path, Path): path = Path(path)
        path = Path(path.as_posix())
        self.results = DSheetPilingResults()
        self.results.load_json(path)


if __name__ == "__main__":

    model_path = os.environ["MODEL_PATH"]  # model_path defined as environment variable
    result_path = r"../../results/example_results.json"
    soil_data = {"Klei": {"soilcohesion": 10}}

    model = DSheetPiling(model_path)
    model.adjust_soil(soil_data)
    model.execute()
    model.save_results(result_path)

    loaded_model = DSheetPiling(model_path)
    loaded_model.load_results(result_path)

    model_check = model.results.__eq__(loaded_model.results)
    if model_check:
        print(f"Were the results loaded correctly?  -->  {model_check} \u2705")
    else:
        print(f"Were the results loaded correctly?  -->  {model_check} \u274C")


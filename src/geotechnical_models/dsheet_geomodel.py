import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict, field
from geomodel import GeoModelBase
from pathlib import Path, WindowsPath
from numpy.typing import NDArray
from typing import Union, List, Tuple, Any, Optional, Annotated, Type
from geolib.models.dsheetpiling import DSheetPilingModel
import json


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

    z: Optional[List[float] | Annotated[NDArray[np.float64], ("n_points")]] = None
    moment: Optional[List[float] | Annotated[NDArray[np.float64], ("n_points")]] = None
    shear: Optional[List[float] | Annotated[NDArray[np.float64], ("n_points")]] = None
    displacement: Optional[List[float] | Annotated[NDArray[np.float64], ("n_points")]] = None
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

    def save_json(self, path: Union[str, WindowsPath]) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f)

    def load_json(self, path: Union[str, WindowsPath]) -> None:
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

    def __init__(self, model_path: str | WindowsPath | Path) -> None:
        super(GeoModelBase, self).__init__()
        self.parse_model(model_path)

    def parse_model(self, model_path: str | WindowsPath | Path) -> None:
        if isinstance(model_path, str):
            model_path = Path(model_path)
        geomodel = DSheetPilingModel()
        geomodel.parse(model_path)
        self.geomodel = geomodel

    def execute(self) -> None:
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
        self.results.save_json(path)

    def load_results(self, path: str | WindowsPath | Path) -> None:
        self.results = DSheetPilingResults()
        self.results.load_json(path)


if __name__ == "__main__":

    model_path = r"C:\Users\mavritsa\Stichting Deltares\Sito-IS 2023 SO Emerging Topics - Moonshot 5 - 02_Asset performance\ARK case study\Geotechnical models\D-Sheet Piling\N60_3_5-060514-red.shi"
    model_path = Path(model_path)
    result_path = Path("../../results/example_results.json")

    model = DSheetPiling(model_path)
    model.execute()
    model.save_results(result_path)

    loaded_model = DSheetPiling(model_path)
    loaded_model.load_results(result_path)

    print("Were the results loaded correctly?  -->  " + str(model.results.__eq__(loaded_model.results)))


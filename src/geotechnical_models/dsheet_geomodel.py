import os
import pandas as pd
from dataclasses import dataclass, asdict, field
from geomodel import GeoModelBase
from pathlib import Path, WindowsPath
from typing import Union, List, Tuple, Any, Optional
from geolib.models.dsheetpiling import DSheetPilingModel
import json


@dataclass
class DSheetPilingStageResults:
    """
    Class representing the results of a D-Sheet Piling stage.
    """

    stage_id: int
    z: list
    moment: list
    shear: list
    displacement: list
    max_moment: float = field(init=False)
    max_shear: float = field(init=False)
    max_displacement: float = field(init=False)

    def __post_init__(self):
        self.max_moment = max(abs(self.moment))
        self.max_shear = max(abs(self.shear))
        self.max_displacement = max(abs(self.displacement))


class DSheetPilingResults:
    """
    Process the results of a D-Sheet Piling model.
    """

    z = None
    moment = None
    shear = None
    displacement = None
    max_moment = None
    max_shear = None
    max_displacement = None
    n_stages = None
    stage_results = None

    def __init__(self):
        pass

    def read(self, stage_results: List[DSheetPilingStageResults]) -> None:
        """
        Read the results of the D-Sheet Piling model and store them in the class.
        """
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
        """
        Convert the results of the D-Sheet Piling model to a dictionary.
        """
        stage_result_dicts = [asdict(stage_result) for stage_result in self.stage_results]
        keys = list(stage_result_dicts[0].keys())
        stage_result_dict = {key: [stage_result_dict[key] for stage_result_dict in stage_result_dicts] for key in keys}
        stage_result_dict["z"] = stage_result_dict["z"][0]
        return stage_result_dict

    def from_dict(self, result_dict) -> None:
        """
        Convert a dictionary to a D-Sheet Piling results object.
        """
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
        """
        Save the results of the D-Sheet Piling model to a JSON file.
        """
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f)

    def load_json(self, path: Union[str, WindowsPath]) -> None:
        """
        Load the results of the D-Sheet Piling model from a JSON file.
        """
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
    """
    Class to execute and extract the results of a D-Sheet Piling model.
    """

    def __init__(self, model_path: Union[str, WindowsPath]) -> None:
        """
        Initialize the D-Sheet Piling model.
        """
        super(GeoModelBase, self).__init__()
        self.parse_model(model_path)

    def parse_model(self, model_path: Union[str, WindowsPath]) -> None:
        """
        Parse the geotechnical model.
        """
        if isinstance(model_path, str):
            model_path = Path(model_path)
        geomodel = DSheetPilingModel()
        geomodel.parse(model_path)
        self.geomodel = geomodel

    def execute(self) -> None:
        """
        Execute the geotechnical model.
        """
        self.geomodel.execute()  # Make sure to add 'geolib.env' in run directory

    def get_results(self) -> DSheetPilingResults:
        """
        Extract the stage results of the geotechnical model.
        """
        if self.geomodel.output is None:
            raise ValueError("The geotechnical model has not been executed.")
        self.stage_results = self.read_dsheet_results()

    def read_dsheet_results(self) -> DSheetPilingResults:
        """
        Read the results of the D-Sheet Piling geotechnical model.
        
        This method extracts and processes the calculation results from the D-Sheet Piling model output.
        It collects data for each construction stage including:
        - Wall points (z-coordinates)
        - Bending moments along the sheet pile
        - Shear forces along the sheet pile
        - Lateral displacements along the sheet pile
        
        Returns:
            DSheetPilingResults: An object containing processed results for all construction stages.
                                Each stage contains arrays of z-coordinates, moments, shear forces,
                                and displacements along the sheet pile wall.
        
        Note:
            TODO: Use the maximum values for each stage from the resume output.
            TODO: Anchor results are not currently processed but will be implemented in a future update.
        """
        stage_max = {
            int(res['stagenumber']):
                {"moment": abs(res['moment']), "shear": abs(res['shearforce']), "disp": abs(res['displacement'])}
            for res in self.geomodel.output.resume.resume
        }

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

        stage_results = DSheetPilingResults()
        stage_results.read(stage_result_lst)

        return stage_results

    def save_results(self, path: Union[str, WindowsPath]) -> None:
        """
        Save the results of the geotechnical model to a JSON file.
        """
        self.stage_results.save_json(path)

    def load_results(self, path: Union[str, WindowsPath]) -> None:
        """
        Load the results of the geotechnical model from a JSON file.
        """
        self.stage_results = DSheetPilingResults()
        self.stage_results.load_json(path)


if __name__ == "__main__":

    model_path = r"C:\Users\mavritsa\Stichting Deltares\Sito-IS 2023 SO Emerging Topics - Moonshot 5 - 02_Asset performance\ARK case study\Geotechnical models\D-Sheet Piling\N60_3_5-060514-red.shi"
    model_path = Path(model_path)
    result_path = Path("../../results/example_results.json")

    model = DSheetPiling(model_path)
    model.execute()
    model.get_results()
    model.save_results(result_path)

    model2 = DSheetPiling(model_path)
    model2.load_results(result_path)
    model2.get_results()
    print("Were the results loaded correctly?  -->  " + str(model.stage_results == model2.stage_results))


import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, asdict, field
from numpy.typing import NDArray
from typing import List, Optional, Annotated


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

    def save_json(self, path: str | Path) -> None:
        if not isinstance(path, Path): path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f)

    def load_json(self, path: str | Path) -> None:
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


if __name__ == "__main__":

    pass


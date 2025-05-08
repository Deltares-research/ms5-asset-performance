import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, asdict, field
from numpy.typing import NDArray
from typing import Optional, Annotated, NamedTuple


class WallProperties(NamedTuple):
    SheetPilingElementMaterialType: float | int
    SheetPilingElementEI: float | int
    SheetPilingElementWidth: float | int
    SheetPilingElementLevel: float | int
    SheetPilingElementHeight: float | int
    SheetPilingPileWidth: float | int
    SheetPilingElementSectionArea: float | int
    SheetPilingElementResistingMoment: float | int
    SheetPilingElementReductionFactorEI: float | int
    SheetPilingElementNote: float | int
    SheetPilingElementMaxCharacteristicMoment: float | int
    SheetPilingElementMaxPlasticCharacteristicMoment: float | int
    SheetPilingElementKMod: float | int
    SheetPilingElementMaterialFactor: float | int
    sSheetPilingElementReductionFactorMaxMoment: float | int
    DiaphragmWallIsSymmetric: float | int
    DiaphragmWallPosEIElastoPlastic1: float | int
    DiaphragmWallNegEIElastoPlastic1: float | int
    DiaphragmWallPosMomElastic: float | int
    DiaphragmWallNegMomElastic: float | int
    DiaphragmWallPosMomPlastic: float | int
    DiaphragmWallNegMomPlastic: float | int
    DiaphragmWallPosEIElastoPlastic2: float | int
    DiaphragmWallPosMomElastoPlastic: float | int
    DiaphragmWallNegEIElastoPlastic2: float | int
    DiaphragmWallNegMomElastoPlastic: float | int
    WoodenSheetPilingElementE: float | int
    WoodenSheetPilingElementCharacFlexuralStrength: float | int
    WoodenSheetPilingElementKSys: float | int
    WoodenSheetPilingElementKDef: float | int
    WoodenSheetPilingElementPsi2Eff: float | int
    WoodenSheetPilingElementMaterialFactor: float | int
    WoodenSheetPilingElementKModFShort: float | int
    WoodenSheetPilingElementKModFLong: float | int
    WoodenSheetPilingElementKModE: float | int


class AnchorProperties(NamedTuple):
    Nr: int
    Level: float
    Emod: float
    Cross_sect: float
    Length: float
    YieldF: float
    Angle: float
    Height: float
    Side: int
    Name: str


class WaterLevel(NamedTuple):
    name: str
    lvl: float
    x_coord: float = 0.
    unknown_entry: int = 2


class WaterData:

    def __init__(self, water_input: str) -> None:
        self.parse(water_input)

    def parse(self, water_input: str) -> None:

        water_lines = water_input.split('\n')

        water_lvls = []
        for line in water_lines[2::4]:
            idx = water_lines.index(line)
            water_lvl = WaterLevel(
                name=water_lines[idx],
                lvl=float(water_lines[idx+1]),
                x_coord=float(water_lines[idx+2]),
            )
            water_lvls.append(water_lvl)

        self.water_lvls = water_lvls

    def adjust(self, new_lvls: dict[str, float]) -> None:

        water_lvl_names = [water_lvl.name for water_lvl in self.water_lvls]
        new_lvl_names = list(new_lvls.keys())

        for new_lvl_name in new_lvl_names:
            if new_lvl_name not in water_lvl_names:
                message = f"New water level name {new_lvl_name} was not found in existing water level names."
                raise AttributeError(message)

        water_lvls = []
        for water_lvl in self.water_lvls:
            if water_lvl.name in new_lvl_names:
                water_lvl = WaterLevel(
                    name=water_lvl.name,
                    lvl=float(new_lvls[water_lvl.name]),
                    x_coord=water_lvl.x_coord,
                )
            water_lvls.append(water_lvl)

        self.water_lvls = water_lvls

    def add(self, new_lvls: dict[str, list[float] | tuple[float, ...]]) -> None:
        for (name, params) in new_lvls.items():
            lvl, x_coord = params
            water_lvl = WaterLevel(name=name, lvl=lvl, x_coord=x_coord)
            self.water_lvls.append(water_lvl)

    def remove(self, removed_lvls: list[str] | tuple[str, ...]) -> None:
        water_lvls = []
        for water_lvl in self.water_lvls:
            if not water_lvl.name in removed_lvls:
                water_lvls.append(water_lvl)
        self.water_lvls = water_lvls

    def write(self) -> str:

        water_lines = [
            f"{len(self.water_lvls)} Number of Waterlevels ",
            "  3 Number of Data per Waterlevel "
        ]

        for water_lvl in self.water_lvls:
            water_line = [
            water_lvl.name,  # Name
            f"{water_lvl.lvl:.2f}",  # Canal lvl
            f"{water_lvl.x_coord:.2f}",  # X-coordinate
            f"{water_lvl.unknown_entry:d}"  # ???
            # TODO: Use these lines in case water level handling does not work.
            # f"      {water_lvl.lvl:.2f}" if water_lvl.lvl >= 0 else f"     {water_lvl.lvl:.2f}",  # Canal lvl
            # f"      {water_lvl.x_coord:.2f}",  # X-coordinate
            # f"         {water_lvl.unknown_entry:d}"  # ???
            ]
            water_lines += water_line

        water_lines = "\n".join(water_lines)

        return water_lines


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

    z: Optional[list[float]] = None
    moment: Optional[list[list[float]]] = None
    shear: Optional[list[list[float]]] = None
    displacement: Optional[list[list[float]]] = None
    max_moment: list[float] = None
    max_shear: list[float] = None
    max_displacement: list[float] = None
    n_stages: int = None
    stage_results = None

    def __init__(self):
        pass

    def __eq__(self, other: "DSheetPilingResults") -> bool:
        return self.__dict__ == other.__dict__

    def read(self, stage_results: list[DSheetPilingStageResults]) -> None:
        self.stage_results = stage_results
        self.n_stages = len(stage_results)
        self.z = [stage_result.z for stage_result in stage_results][0]
        self.moment = [stage_result.moment for stage_result in stage_results]
        self.shear = [stage_result.shear for stage_result in stage_results]
        self.displacement = [stage_result.displacement for stage_result in stage_results]
        self.max_moment = [stage_result.max_moment for stage_result in stage_results]
        self.max_shear = [stage_result.max_shear for stage_result in stage_results]
        self.max_displacement = [stage_result.max_displacement for stage_result in stage_results]

    def to_dict(self) -> dict[str, list[float | int]]:
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
        path = Path(path.as_posix())
        path_folder = path.parent
        if not path_folder.exists():
            raise NotADirectoryError("Result path does not exist.")
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f)

    def load_json(self, path: str | Path) -> None:
        if not isinstance(path, Path): path = Path(path)
        path = Path(path.as_posix())
        path_folder = path.parent
        if not path_folder.exists():
            raise NotADirectoryError("Result path does not exist.")
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


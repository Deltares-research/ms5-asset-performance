"""
D-SheetPiling limit state function assembly.

Builds an LSF callable from a DSheetPiling model, a GaussianState,
and a performance configuration. Uses a payload dict to communicate
parameter updates to the model.

Payload format::

    {
        "soil":   {"Sand": {"soilphi": 30.0}, "Clay": {"soilcohesion": 20.0}},
        "water":  {"WL_left": -1.5},
        "loads":  {"Surcharge": (10.0, 0.0)},
        "anchor": {"Level": -2.0},
        "wall":   {"corrosion": 0.3, "start_thickness": 9.5},
    }

Each key is optional. Only present keys trigger the corresponding update.
"""

from typing import Type, Tuple, Callable, Dict, List, Optional, Any
import numpy as np
from numpy.typing import NDArray

from src.geotechnical_models.dsheetpiling.model import DSheetPiling
from src.rvs.state import StateBase
from src.ptk.lsf import build_lsf, LSFType
from .params import (
    unpack_soil_params,
    unpack_water_params,
    unpack_load_params,
    unpack_anchor_params,
)


def build_payload(
    params: Dict[str, float],
    geomodel: DSheetPiling,
) -> Dict[str, Any]:
    """Build a payload dict from a flat parameter dict.

    Applies the naming conventions (SoilName_paramName, water_LevelName, etc.)
    to route each parameter to the correct model update function.

    Args:
        params: Flat dict of parameter values, e.g.
            {"Sand_soilphi": 30.0, "water_WL_left": -1.5, "corrosion": 0.3}.
        geomodel: DSheetPiling model (used to get valid soil/water names).

    Returns:
        Payload dict with keys: soil, water, loads, anchor, wall.
        Only keys with non-empty data are included.
    """
    payload = {}

    soil = unpack_soil_params(params, list(geomodel.soils.keys()))
    if soil:
        payload["soil"] = soil

    water = unpack_water_params(params, [lvl.name for lvl in geomodel.water.water_lvls])
    if water:
        payload["water"] = water

    load_names = list(geomodel.uniform_loads.keys()) if geomodel.uniform_loads else []
    loads = unpack_load_params(params, load_names)
    if loads:
        payload["loads"] = loads

    # Wall / corrosion: look for "corrosion" key
    if "corrosion" in params:
        payload["wall"] = {
            "corrosion": params["corrosion"],
            "start_thickness": params.get("start_thickness", 9.5),
        }

    return payload


def apply_payload(geomodel: DSheetPiling, payload: Dict[str, Any]) -> None:
    """Apply a payload dict to the D-SheetPiling model.

    Only calls update functions for keys present in the payload.

    Args:
        geomodel: DSheetPiling model to update.
        payload: Payload dict with optional keys: soil, water, loads, anchor, wall.
    """
    if "soil" in payload:
        geomodel.update_soils(payload["soil"])

    if "water" in payload:
        geomodel.update_water(payload["water"])

    if "loads" in payload:
        geomodel.update_uniform_loads(payload["loads"])

    if "anchor" in payload:
        anchor_txt = geomodel.geomodel.input.input_data.anchors
        updated_txt = unpack_anchor_params(payload["anchor"], anchor_txt)
        geomodel.update_anchors(updated_txt)

    if "wall" in payload:
        wall = payload["wall"]
        if "corrosion" in wall:
            geomodel.apply_corrosion(
                wall["corrosion"], wall.get("start_thickness", 9.5)
            )


def safety_fn(
    params: Dict[str, float],
    geomodel: DSheetPiling,
    state: Type[StateBase],
    performance_config: Tuple[str, Callable],
    standardized_rv: bool = False,
) -> float:
    """Compute safety factor by running the D-SheetPiling model.

    Builds a payload from the flat params dict, applies it to the model,
    executes, and evaluates the performance function.

    Args:
        params: Flat dict of parameter values keyed by variable name.
        geomodel: D-SheetPiling model wrapper.
        state: Gaussian state with variable definitions.
        performance_config: Tuple of (measure_name, performance_fn).
        standardized_rv: If True, transform from standard normal domain.

    Returns:
        Safety factor (>1 = safe).
    """
    rvs = {key: val for key, val in params.items() if key in state.names}
    if standardized_rv:
        x_st = np.asarray(list(rvs.values()))
        x = state.transform(x_st)
        rvs = {key: val for key, val in zip(rvs.keys(), x)}

    # Merge transformed RVs with any extra params (e.g. corrosion)
    all_params = {**rvs, **{k: v for k, v in params.items() if k not in state.names}}

    payload = build_payload(all_params, geomodel)
    apply_payload(geomodel, payload)
    geomodel.execute()

    measure_name, performance_fn = performance_config
    measure = getattr(geomodel.results, measure_name)
    return performance_fn(measure)


def package_lsf(
    geomodel: DSheetPiling,
    state: Type[StateBase],
    performance_config: Tuple[str, Callable],
    standardized_rv: bool = False,
) -> LSFType:
    """Create an LSF callable from a D-SheetPiling model and state.

    Args:
        geomodel: Parsed D-SheetPiling model.
        state: Gaussian state with variable definitions.
        performance_config: Tuple of (measure_name, performance_fn).
        standardized_rv: If True, LSF receives standard normal values.

    Returns:
        LSF callable with positional args matching state.names.
    """
    if not hasattr(geomodel, "geomodel"):
        raise ValueError("Geotechnical model has not yet been parsed.")

    lsf = build_lsf(
        state.names,
        lambda x: safety_fn(x, geomodel, state, performance_config, standardized_rv),
    )
    return lsf

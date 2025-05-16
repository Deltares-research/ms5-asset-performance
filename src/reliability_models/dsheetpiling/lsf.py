from src.geotechnical_models.dsheetpiling.model import *
from src.rvs.state import *
from typing import Type, Tuple, Dict, Callable


LSFInputType = Annotated[Tuple[float,...] | List[float] | NDArray[np.float64], "lsf_input"]
LSFType = Callable[[LSFInputType], float]


def build_lsf(arg_names, body_func):
    """
    Creates a new function with named arguments, that internally calls `body_func(**kwargs)`.
    """
    args_str = ", ".join(arg_names)
    dict_pack = ", ".join([f"'{arg}': {arg}" for arg in arg_names])

    func_code = f"""
def lsf({args_str}):
    params = {{{dict_pack}}}
    return body_func(params) - 1
"""
    namespace = {'body_func': body_func}
    exec(func_code, namespace)
    fn = namespace['lsf']
    fn._generated_source = func_code
    return fn


def unpack_soil_params(params: Dict[str, float], soil_layers: List[str]) -> Dict[str, Dict[str, float]]:
    soil_data = {}
    for (key, val) in params.items():
        try:
            soil_name = key.split("_")[0]
            param_name = key.split("_")[1]
        except:
            continue
        if not soil_name in soil_layers:
            continue
        param_value = val
        if not soil_name in list(soil_data.keys()):
            soil_data[soil_name] = {param_name: float(param_value)}
        else:
            if not param_name in list(soil_data[soil_name].keys()):
                soil_data[soil_name][param_name] = float(param_value)
    return soil_data


def unpack_water_params(params: Dict[str, float], water_lvls: List[str]) -> Dict[str, float]:
    water_data = {}
    for (key, val) in params.items():
        try:
            water_lvl_name = key.split("_")[-1]
        except:
            continue
        if not water_lvl_name in water_lvls:
            continue
        water_data[water_lvl_name] = float(val)
    return water_data

def unpack_wall_data(params: Dict[str, float], wall_dict: Dict[str, float]) -> Dict[str, float]:
    wall_names = list(wall_dict.keys())
    for (key, val) in params.items():
        name = key.split("_")[-1]
        if not name in wall_names:
            continue
        wall_dict[name] = float(val)
    return wall_dict

def safety_fn(
        params: Dict[str, float],
        geomodel: DSheetPiling,
        state: Type[StateBase],
        performance_config: Tuple[str, Callable[[float | List[float]], float]],
        standardized_rv: bool = False
) -> float:

    """
    NOTE: Soil layer name and soil parameter name must be split by a lowerdash("_").
    TODO: Use something less common that _?
    """

    rvs = {key: param for (key, param) in params.items() if key in state.names}
    if standardized_rv:
        x_st = np.asarray(list(rvs.values()))
        x = state.transform(x_st)
        rvs = {key: val for (key, val) in zip(rvs.keys(), x)}

    soil_data = unpack_soil_params(rvs, list(geomodel.soils.keys()))
    water_data = unpack_water_params(rvs, [lvl.name for lvl in geomodel.water.water_lvls])

    geomodel.update_soils(soil_data)
    geomodel.update_water(water_data)
    geomodel.execute()
    results = geomodel.results

    measure_name, performance_fn = performance_config
    measure = getattr(results, measure_name)
    sf = performance_fn(measure)

    return sf


def package_lsf(
        geomodel: DSheetPiling,
        state: Type[StateBase],
        performance_config: Tuple[str, Callable[[float | List[float]], float]],
        standardized_rv: bool = False
) -> LSFType:

    model_parsed = hasattr(geomodel, "geomodel")
    if not model_parsed:
        raise ValueError("Geotechnical model has not yet been parsed.")

    lsf = build_lsf(state.names, lambda x: safety_fn(x, geomodel, state, performance_config, standardized_rv))

    return lsf


if __name__ == "__main__":

    pass


import numpy as np
from numpy.typing import NDArray
from src.geotechnical_models.dsheetpiling.model import *
from src.rvs.state import *
from typing import Optional, Type, Tuple, Dict, Callable


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

    # """ Check is soil layers and parameters exist in model """
    # geomodel_soil_names = geomodel.get_soils()
    # for soil_name, soil_params in soil_layers.items():
    #     if soil_name not in list(geomodel_soil_names.keys()):
    #         raise ValueError(f"Soil name {soil_name} not found in geomodel.")
    #     geomodel_soil = geomodel_soil_names[soil_name]
    #     geomodel_layer_params = list(vars(geomodel_soil).keys())
    #     for soil_param in soil_params:
    #         if soil_param not in geomodel_layer_params:
    #             raise ValueError(f"Soil parameter {soil_param} of soil {soil_name} not found in geomodel.")

    lsf = build_lsf(state.names, lambda x: safety_fn(x, geomodel, state, performance_config, standardized_rv))

    return lsf


if __name__ == "__main__":

    geomodel_path = os.environ["MODEL_PATH"]
    form_path = os.environ["FORM_PATH"]
    geomodel = DSheetPiling(geomodel_path, form_path)

    state = GaussianState(rvs=[
        MvnRV(mus=[30, 10], stds=[3, 1], names=["Klei_soilphi", "Klei_soilcohesion"]),
        MvnRV(mus=[1], stds=[0.1], names=["water_A"])
    ])
    performance_config = ("max_moment", lambda x: 150. / (x[0] + 1e-5))

    """ The args are "soilphi", "soilcohesion" and "water_A" respectively. """
    lsf = package_lsf(geomodel, state, performance_config, False)
    limit_state = lsf(30, 10, 1)

    lsf_st = package_lsf(geomodel, state, performance_config, True)
    limit_state_st = lsf_st(0, 0, 0)


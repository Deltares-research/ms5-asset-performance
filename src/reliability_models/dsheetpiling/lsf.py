import numpy as np
from numpy.typing import NDArray
from src.geotechnical_models.dsheetpiling.model import *
from src.rvs.state import *
from typing import Optional, Type, Tuple, Dict, Callable
import itertools
import inspect


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
    return namespace['lsf']


def unpack_params(params: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    soil_data = {}
    for (key, val) in params.items():
        soil_name = key.split("_")[0]
        param_name = key.split("_")[1]
        param_value = val
        if not soil_name in list(soil_data.keys()):
            soil_data[soil_name] = {param_name: float(param_value)}
        else:
            if not param_name in list(soil_data[soil_name].keys()):
                soil_data[soil_name][param_name] = float(param_value)
    return soil_data


def safety_fn(
        params: Dict[str, float],
        geomodel: DSheetPiling,
        rv: Type[RV],
        performance_config: Tuple[str, Callable[[float | List[float]], float]],
        standardized_rv: bool = False
) -> float:

    """
    NOTE: Soil layer name and soil parameter name must be split by a lowerdash("_").
    TODO: Use something less common that _?
    """

    if standardized_rv:
        x_st = np.asarray(list(params.values()))
        x = rv.transform(x_st)
        params = {key: val for (key, val) in zip(params.keys(), x)}

    soil_data = unpack_params(params)
    geomodel.update_soils(soil_data)
    geomodel.execute()
    results = geomodel.results

    measure_name, performance_fn = performance_config
    measure = getattr(results, measure_name)
    sf = performance_fn(measure)

    return sf


def package_lsf(
        geomodel: DSheetPiling,
        soil_layers: Dict[str, Tuple[str, ...]],
        rv: Type[RV],
        performance_config: Tuple[str, Callable[[float | List[float]], float]],
        standardized_rv: bool = False
) -> LSFType:

    model_parsed = hasattr(geomodel, "geomodel")
    if not model_parsed:
        raise ValueError("Geotechnical model has not yet been parsed.")

    """ Check is soil layers and parameters exist in model """
    geomodel_soil_names = geomodel.get_soils()
    for soil_name, soil_params in soil_layers.items():
        if soil_name not in list(geomodel_soil_names.keys()):
            raise ValueError(f"Soil name {soil_name} not found in geomodel.")
        geomodel_soil = geomodel_soil_names[soil_name]
        geomodel_layer_params = list(vars(geomodel_soil).keys())
        for soil_param in soil_params:
            if soil_param not in geomodel_layer_params:
                raise ValueError(f"Soil parameter {soil_param} of soil {soil_name} not found in geomodel.")

    rvs = [[name+"_"+param for param in soil_layers[name]] for name in soil_layers.keys()]
    rvs = list(itertools.chain(*rvs))

    lsf = build_lsf(rvs, lambda x: safety_fn(x, geomodel, rv, performance_config, standardized_rv))

    return lsf


if __name__ == "__main__":

    geomodel_path = os.environ["MODEL_PATH"]
    form_path = os.environ["FORM_PATH"]
    geomodel = DSheetPiling(geomodel_path, form_path)

    soil_layers = {"Klei": ("soilphi", "soilcohesion")}
    state = GaussianState(rvs=[MvnRV(mus=[30, 10], stds=[3, 1], names=["Klei_soilphi", "Klei_soilcohesion"])])
    performance_config = ("max_moment", lambda x: 150. / (x[0] + 1e-5))

    """ The first arg is "soilphi" and the second is "soilcohesion" for "Klei", as declared. """
    lsf = package_lsf(geomodel, soil_layers, state.rvs[0], performance_config, False)
    limit_state = lsf(30, 10)

    lsf = package_lsf(geomodel, soil_layers, state.rvs[0], performance_config, True)
    limit_state_st = lsf(0, 0)


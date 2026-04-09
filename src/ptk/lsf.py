"""
Generic LSF (limit state function) builder.

Creates callable LSFs with named arguments from a body function.
"""

from typing import Callable, List, Tuple, Annotated
import numpy as np
from numpy.typing import NDArray


LSFType = Callable[..., float]


def build_lsf(arg_names: List[str], body_func: Callable) -> LSFType:
    """Create a function with named positional arguments that calls body_func.

    The generated function packs its arguments into a dict and passes it
    to body_func. The return value is ``body_func(params) - 1`` (safety
    factor convention: g > 0 = safe).

    Args:
        arg_names: List of argument names for the generated function.
        body_func: Callable that takes a dict and returns a safety factor.

    Returns:
        A callable ``lsf(x0, x1, ...) -> float``.
    """
    args_str = ", ".join(arg_names)
    dict_pack = ", ".join([f"'{arg}': {arg}" for arg in arg_names])

    func_code = f"""
def lsf({args_str}):
    params = {{{dict_pack}}}
    return body_func(params) - 1
"""
    namespace = {"body_func": body_func}
    exec(func_code, namespace)
    fn = namespace["lsf"]
    fn._generated_source = func_code
    return fn

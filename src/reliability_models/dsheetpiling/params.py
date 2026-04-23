"""
D-SheetPiling parameter unpacking utilities.

Converts flat parameter dicts (keyed by "SoilName_paramName" convention)
into structured dicts suitable for DSheetPiling.update_soils(), etc.
"""

from typing import Dict, List


def unpack_soil_params(params: Dict[str, float], soil_layers: List[str]) -> Dict[str, Dict[str, float]]:
    """Unpack flat params dict into per-soil-layer param dicts.

    Convention: "SoilName_paramName" → {"SoilName": {"paramName": value}}.

    Args:
        params: Flat dict, e.g. {"Sand_soilphi": 30, "Clay_soilcohesion": 20}.
        soil_layers: List of valid soil layer names.

    Returns:
        Nested dict, e.g. {"Sand": {"soilphi": 30}, "Clay": {"soilcohesion": 20}}.
    """
    soil_data = {}
    for key, val in params.items():
        parts = key.split("_")
        if len(parts) < 2:
            continue
        soil_name = parts[0]
        param_name = parts[1]
        if soil_name not in soil_layers:
            continue
        if soil_name not in soil_data:
            soil_data[soil_name] = {}
        if param_name not in soil_data[soil_name]:
            soil_data[soil_name][param_name] = float(val)
    return soil_data


def unpack_water_params(params: Dict[str, float], water_lvls: List[str]) -> Dict[str, float]:
    """Unpack flat params dict into water level values.

    Convention: "water_LevelName" → {"LevelName": value}.

    Args:
        params: Flat dict, e.g. {"water_WL_left": -1.5}.
        water_lvls: List of valid water level names.

    Returns:
        Dict, e.g. {"WL_left": -1.5}.
    """
    water_data = {}
    for key, val in params.items():
        if key not in water_lvls:
            continue
        water_data[key] = float(val)
    return water_data


def unpack_load_params(params: Dict[str, float], load_names: List[str]) -> Dict[str, tuple]:
    """Unpack flat params dict into uniform load values.

    Convention: "LoadName_left" or "LoadName_right" → {"LoadName": (left, right)}.

    Args:
        params: Flat dict.
        load_names: List of valid load names.

    Returns:
        Dict mapping load name to (left_value, right_value) tuple.
    """
    load_data = {}
    for key, val in params.items():
        parts = key.split("_")
        if len(parts) < 2:
            continue
        load_name = key
        if load_name not in load_names:
            continue
        load_side = parts[-1]
        if load_side == "left":
            load_data[load_name] = (float(val), 0)
        elif load_side == "right":
            load_data[load_name] = (0, float(val))
    return load_data


def unpack_anchor_params(params: Dict[str, float], anchor_txt: str) -> str:
    """Update anchor text string with values from params dict.

    Convention: "anchor_FieldName" updates the corresponding field.

    Args:
        params: Flat dict, e.g. {"anchor_Level": -2.0}.
        anchor_txt: Raw anchor text from D-SheetPiling input file.

    Returns:
        Updated anchor text string.
    """
    lines = anchor_txt.splitlines()
    data_values = lines[-1].split()

    field_map = {
        "Nr": 0, "Level": 1, "E-mod": 2, "Cross": 3,
        "Length": 4, "YieldF": 5, "Angle": 6,
        "Height": 7, "Side": 8,
    }

    for key, val in params.items():
        parts = key.split("_")
        if parts[0].lower() != "anchor":
            continue
        field = parts[-1]
        if field in field_map:
            data_values[field_map[field]] = f"{val:.2f}"

    lines[-1] = (
        f"  {data_values[0]}"
        f"  {data_values[1]:>5s}"
        f"  {data_values[2]:>11s}"
        f"  {data_values[3]:>11s}"
        f"    {data_values[4]:>5s}"
        f" {data_values[5]:>8s}"
        f"    {data_values[6]:>5s}"
        f"     {data_values[7]:>4s}"
        f"      {data_values[8]}"
        f" {data_values[9]}"
    )

    return "\n".join(lines)

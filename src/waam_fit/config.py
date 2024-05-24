import tomllib, os
from typing import Any

class ConfigError(Exception):
    pass

with open(os.path.dirname(os.path.abspath(__file__)) + "/WAAM.toml", "rb") as file:
    config = tomllib.load(file)
# Validate config file
for feature in config["features"]:
    try:
        filter_list = config["filter"]
    except:
        filter_list  = []
    try:
        filter = config["features"][feature]["filter"]
    except:
        filter = None
    if not (filter is None or filter in filter_list):
        raise ConfigError("Use of unspecified filter: " + str(filter))
    
    try:
        style_list = config["styles"]
    except:
        style_list = []
    try:
        style = config["features"][feature]["style"]
    except:
        style = None
    if not (style is None or style in style_list):
        raise ConfigError("Did not find style " + str(style))
    
INCLUDEBASEPLATE = config.get("settings", {}).get("includeBaseplate", False)

ANALYSIS_DATATYPES = ["radii.inner", "radii.outer", 
                                 "gradients.inner", "gradients.outer", "gradients.inner_deviation", "gradients.inner_tan", 
                                 "distances.inner", "distances.outer", 
                                 "angles.inner", "angles.outer",
                                 "heights", "tilt_angles"]

def __style_from_config__(style_key: str) -> dict[str, Any]:
    try:
        style = config["styles"][style_key]
    except:
        style = {}
    return style

def __constraints_from_config__(group, feature):
    constraints = {}
    for limit in ["max", "min"]:
        try:
            constraints[limit] = config["constraints"][group][feature][limit]
        except:
            constraints[limit] = None
    return constraints

def __verify_datatype__(datatype: str):
    if (datatype == "") or (datatype is None):
        raise ConfigError("Missing data")
    elif datatype not in ANALYSIS_DATATYPES:
        raise ConfigError("Invalid data was specified: " + str(datatype))
    else:
        return datatype

def __parse_name__(name: str):
    if not (name == "" or name is None):
        group, _, name =  name.rpartition("/")
        return group, name
    else:
        raise ConfigError("Missing name attribute")

import gmsh
import os
from math import nan
import numpy as np
from waam_fit import config

def plot_in_gmsh(elements, results):
    """Visualize provided data in gmsh by respecting options set in WAAM.toml
    Results are added as groups to gmsh as provided in subdicts

    Args:
        elements (list): gmsh elements on which results shall be applied
        results (dict[str, dict[str, array]]): Hierarchial presentation of results
    """
    elements = np.array(elements)
    for feature in config.config["features"].values():
        group, name = config.__parse_name__(feature["name"])
        data_key = config.__verify_datatype__((feature["data"]))
        scale = feature["scale"] if "scale" in feature else 1.0
        try: 
            data = results[data_key] * scale
        except:
            print(f"Error\t: Did not find {data_key} in computed results") 
            log_data = [results.items()]
            log_header = " ".join([results.keys()])
            np.savetxt("log_results.csv", log_data, header=log_header)
            
        else:
            print("Info\t: View " + feature['name'] + " was added")
            filter = __get_filter_as_configured__(results, feature)
            try:
                view = __add_as_view_to_gmsh__(elements[filter].tolist(), data[filter].tolist(), name, group)
    
                style = config.__style_from_config__(feature["style"]) if "style" in feature else {}
                max = feature["max"] if "max" in feature else np.max(data[filter])
                min = feature["min"] if "min" in feature else 0
                __set_view_options__(view, max, min, config=style)
            except Exception as error:
                print(error)
                continue
    # Hide mesh
    gmsh.option.set_number("Mesh.SurfaceEdges", 0)
    gmsh.option.set_number("Mesh.VolumeEdges", 0)

    gmsh.fltk.update()

def show():
    while gmsh.fltk.is_available():
        gmsh.fltk.wait()
    gmsh.finalize()

def save_all_views(path):
    if not os.path.exists(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))
    for view in gmsh.view.get_tags():
        name = gmsh.view.option.get_string(view, "Group").replace('/', '_') + '_' + gmsh.view.option.get_string(view, "Name")
        gmsh.view.write(view, path + name + ".msh")

def __add_point_cloud__():
    import random
    t1 = gmsh.view.add("A list-based view")
    n = 1000
    data = []
    for _ in range(n):
        x = random.uniform(0.0,1.0)
        y = random.uniform(0.0,1.0)
        z = 0
        vx = random.uniform(0.0,1.0)
        vy = random.uniform(0.0,1.0)
        vz = random.uniform(0.0,1.0)
        data.extend([x, y, z, vx, vy, vz])

    gmsh.view.add_list_data(t1, "VP", n, data)

def __add_as_view_to_gmsh__(tags, data: list, view_name, group=None) -> int:
    """Add provided `data` as a view for `tags` to gmsh with `view_name` in optional `group`
    
    Returns:
        int: view_tag
    """
    view = gmsh.view.add(view_name)
    if len(data) > 0:
        gmsh.view.add_homogeneous_model_data(view, 0, "", "ElementData", tags=tags, data=data, numComponents=1)
    else:
        print("Info\t: View " + view_name + " is empty")
    gmsh.view.option.set_number(view, "Visible", 0)
    if isinstance(group, str):
        gmsh.view.option.set_string(view, "Group", group)
    if "default" in config.config["styles"]:
        __set_view_options__(view, config=config.config["styles"]["default"])
    return view

def __get_filter_as_configured__(results: dict[str, np.ndarray], feature: dict) -> np.ndarray:
    try:
        filter = feature["filter"]
    except:
        filter = None
    if filter is not None:
        filter_data = results[config.__verify_datatype__(config.config["filter"][filter]["data"])]
        try:
            filter_min = config.config["filter"][filter]["greater_eq"]
        except:
            filter_min = 0
        try:
            filter_max = config.config["filter"][filter]["less_eq"]
        except:
            filter_max = nan
        if filter_min > filter_max:
            filter = (filter_data >= filter_min) + (filter_data <= filter_max)
        elif filter_min <= filter_max:
            filter = (filter_data >= filter_min) * (filter_data <= filter_max)
        return filter
    else:
        data = results[config.__verify_datatype__(feature["data"])]
        return np.ones_like(data, dtype=bool)

def __set_view_options__(view, max = None, min = None, config = {}) -> int:
    if isinstance(max, (int, float)):
        gmsh.view.option.set_number(view, "CustomMax", max)
    if isinstance(min, (int, float)):
        gmsh.view.option.set_number(view, "CustomMin", min)
    
    for (key, value) in config.items():
        if isinstance(value, (int, float)):
            try: 
                gmsh.view.option.set_number(view, key ,value)
            except Exception as error:
                print(error)
        elif isinstance(value, (str)):
            try:
                gmsh.view.option.set_string(view, key, value)
            except Exception as error: 
                print(error)
        elif isinstance(value, tuple) and len(value) == 4:
            try:
                gmsh.view.option.set_color(view, key, r=value[0], g = value[1], b = value[2], a = value[3])
            except Exception as error:
                print(error)

    return view
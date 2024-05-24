from dataclasses import dataclass
from math import nan
import gmsh
import numpy as np
import numpy.typing as npt
import tomllib, os
from typing import Any, Tuple
from waam_fit import rust_methods # type: ignore
from waam_fit import visualize as vis
from waam_fit import config

@dataclass
class Brep:
    node_coordinates: npt.NDArray
    inz: npt.NDArray
    centers: npt.NDArray
    normals: npt.NDArray
    element_tags: npt.NDArray

gmsh.initialize()

def evaluateGeometry(input: str, output:str, triangulationSizing=0.0, base_points: Tuple[Tuple[float, float, float], Tuple[float, float, float]] | None = None) -> None:
    """Evaluate all inner and outer spheres on given shape and save to output

    Args:
        input (string): Path to input shape
        output (string): Path to output shape
        triangulationSizing (float, optional): controls size of triangulation. Defaults to 0.0 for auto-sizing.
    """
    if config.INCLUDEBASEPLATE:
        if base_points is not None:
            print("Info\t: Baseplate defined by " + str(base_points[0]) + " and " + str(base_points[1])) 
        else:
            print("Warning\t: No baseplate was specified.")

    # nc, inz, centers, normals, elementTags = getTriangulation(input, triangulationSizing)
    geometry = getTriangulation(input, triangulationSizing)

    results = compute_indicators(geometry, base_points)

    r_inner = results["radii.inner"]
    gradient = np.zeros_like(r_inner)
    inz = geometry.inz
    centers = geometry.centers
    for i in range(gradient.shape[0]):
        neighbours = (np.isin(inz, inz[i]).sum(axis=1) == 2).nonzero()[0]
        neighbours = neighbours[r_inner[neighbours] != -1]  # neglect invalid radii
        neighbours = neighbours[r_inner[neighbours] != np.Inf]  # neglect infinte radii
        gradient[i] = np.linalg.norm(
            (r_inner[i] - r_inner[neighbours]) / np.linalg.norm(centers[i] - centers[neighbours], axis=1))

    gradient_tan = np.tan(np.deg2rad(results["angles.inner"]/2))
    gradient_deviation = gradient - gradient_tan

    results["gradients.inner"] = gradient
    results["gradients.inner_tan"] = gradient_tan
    results["gradients.inner_deviation"] = gradient_deviation
    
    plot_in_gmsh(geometry.element_tags.tolist(), results)

    vis.save_all_views(output)
    
    vis.show()

def compute_indicators(geometry: Brep, base_points: Tuple[Tuple[float, float, float], Tuple[float, float, float]] | None = None):
    results = {}

    for dir in ["inner", "outer"]:
        dir_fac = -1 if dir == "inner" else 1
        if config.INCLUDEBASEPLATE:
            base_points = base_points if base_points is not None else ((0,0,0), (1,1,1))
            radii, distances, angles, heights, tilt_angles = rust_methods.get_sphere_radii(geometry.centers, dir_fac*geometry.normals, geometry.element_tags.tolist(), base_points)
            results["radii." + dir] = np.array(radii)
            results["distances." + dir] = np.array(distances)
            results["angles." + dir] = np.array(angles)
            if dir == "inner":
                results["heights"] = np.array(heights)
                results["tilt_angles"] = np.array(tilt_angles)
        else:
            radii, distances, angles = rust_methods.get_sphere_radii(geometry.centers, dir_fac*geometry.normals, geometry.element_tags.tolist())
            results["radii." + dir] = np.array(radii)
            results["distances." + dir] = np.array(distances)
            results["angles." + dir] = np.array(angles)
    
    return results

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

def getTriangulation(input: str, triangulationSizing=0.0) -> Brep:
    """Create triangulation mesh on input file and return mesh

    The mesh will be returned as BREP (boundary representation) with the node coordinates, the inzidenz_matrix (which gives the relation between nodes and edges) and ?
    Args:
        input (string): file name with path (either .step/.stp or .stl are supported)
        triangulationSizing (float, optional): Controls mesh-sizing. Defaults to 0.0 for auto-sizing.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: node_coordinates, inzidenz_matrix, vector?
    """
    file_extension = input.split('.')[-1]
    if file_extension in ['stp', 'step']:
        gmsh.model.occ.importShapes(input)
        gmsh.model.occ.synchronize()
        if triangulationSizing != 0:
            gmsh.model.mesh.setSize(
                # Pass mesh sizing trough to points (entities with dimension 0)
                gmsh.model.occ.getEntities(0), triangulationSizing)
        gmsh.model.mesh.generate(2)
        nc, inz, C, N, elemTags = __MshFromGmsh__()
    else:
        raise ValueError('File format is not supported')
    
    geometry = Brep(nc, inz, C, N, elemTags)
    return geometry

def __MshFromGmsh__():
    """Do unknown operation

    Returns:
        tuple: nodeCoordinates(np.ndarray), elementNodeTags(np.ndarray), ?(np.ndarray)
    """
    nodeCoords = np.array([], dtype=float)
    gmsh.model.mesh.renumber_nodes()
    for i in range(0, 3):
        rnodeCoords = gmsh.model.mesh.getNodes(i)[1]
        nodeCoords = np.append(nodeCoords, rnodeCoords)
    nc = nodeCoords.reshape(-1, 3)
    _, elemTags, elemNodeTags = gmsh.model.mesh.getElements(2)
    elemTags = elemTags[0].astype(int)  # type: ignore
    inz = elemNodeTags[0].astype(int).reshape(  # type: ignore
        elemTags.shape[0], -1) - 1

    C = np.mean(nc[inz], axis=1)
    faceIDs = np.zeros(inz.shape[0], dtype=int)
    for entity in gmsh.model.getEntities(2):
        ID = entity[1]
        elemTagsOnFace = gmsh.model.mesh.getElements(2, ID)[1][0]
        for tag in elemTagsOnFace:
            faceIDs[elemTags == tag] = ID

    N = np.cross(nc[inz[:, 1]] - nc[inz[:, 0]], nc[inz[:, 2]] - nc[inz[:, 0]])
    for i in range(inz.shape[0]):
        para = gmsh.model.getParametrization(2, faceIDs[i], C[i])
        N[i] = gmsh.model.getNormal(faceIDs[i], para)
        N[i] = N[i] / np.linalg.norm(N[i])

    return nc, inz, C, N, elemTags


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
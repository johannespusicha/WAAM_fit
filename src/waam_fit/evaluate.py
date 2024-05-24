from dataclasses import dataclass
import gmsh
import numpy as np
import numpy.typing as npt
from typing import Tuple
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
    
    vis.plot_in_gmsh(geometry.element_tags.tolist(), results)

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

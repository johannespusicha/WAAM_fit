import gmsh
import numpy as np
import tomllib, os
from typing import Tuple
import rust_methods

class ConfigError(Exception):
    pass

with open("WAAM.toml", "rb") as file:
    config = tomllib.load(file)
# Validate config file
for value in config["visualization"].values():
    if not value in config["styles"]:
        raise ConfigError("Did not find style " + str(value))

gmsh.initialize()

def evaluateSpheres(input: str, output:str, triangulationSizing=0.0) -> None:
    """Evaluate all inner and outer spheres on given shape and save to output

    Args:
        input (string): Path to input shape
        output (string): Path to output shape
        triangulationSizing (float, optional): controls size of triangulation. Defaults to 0.0 for auto-sizing.
    """
    nc, inz, centers, normals, elementTags = getTriangulation(input, triangulationSizing)

    r_inner, alpha_inner = rust_methods.get_sphere_radii(centers, -normals, elementTags.tolist()) # type: ignore
    r_inner = np.array(r_inner)
    r_outer, alpha_outer = rust_methods.get_sphere_radii(centers, normals, elementTags.tolist()) # type: ignore
    r_outer = np.array(r_outer)

    gradient = np.zeros_like(r_inner)
    for i in range(gradient.shape[0]):
        neighbours = (np.isin(inz, inz[i]).sum(axis=1) == 2).nonzero()[0]
        neighbours = neighbours[r_inner[neighbours] != -1]  # neglect invalid radii
        gradient[i] = np.linalg.norm(
            (r_inner[i] - r_inner[neighbours]) / np.linalg.norm(centers[neighbours] - centers[i], axis=1))
    
    grad_inner_scaled = gradient - gradient.min()

    btm_95_percent = (grad_inner_scaled < grad_inner_scaled.max() * 0.95)
    grad_inner_scaled[grad_inner_scaled >= grad_inner_scaled.max() * 0.95] = grad_inner_scaled[btm_95_percent.nonzero()[0][grad_inner_scaled[btm_95_percent].argmax()]]
    grad_inner_scaled = grad_inner_scaled / grad_inner_scaled.max()

    # Save and show data in gmsh GUI:
    views = []
    v = gmsh.view.add("Normals")
    gmsh.view.addModelData(v, 0, "", "ElementData", elementTags.tolist(), normals.tolist(), numComponents=3)
    views.append(v)
    views.append(__add_as_view_to_gmsh__(elementTags.tolist(), r_inner.tolist(), "Inner Radii")) # type: ignore
    views.append(__add_as_view_to_gmsh__(elementTags.tolist(), alpha_inner, "Inner Angles")) # type: ignore
    views.append(__add_as_view_to_gmsh__(elementTags.tolist(), grad_inner_scaled.tolist(), "Inner Radii Gradients")) # type: ignore
    views.append(__add_as_view_to_gmsh__(elementTags.tolist(), r_outer.tolist(), "Outer Radii")) # type: ignore

    if not os.path.exists(os.path.dirname(output)):
        os.mkdir(os.path.dirname(output))

    for v in views:
        # Set a green to blue color map as the ColorTable = {Green, Red} option is not yet available in API
        gmsh.view.option.set_number(v, "ColormapNumber", 17)
        gmsh.view.option.set_number(v, "ColormapSwap", 1)
        
        # Save views:
        file_names = ["radii_scaled", "gradient_scaled"]
        gmsh.view.write(v, output + file_names.pop() + ".msh")

    # Show last view by default:
    gmsh.view.option.set_number(views[-1], "Visible", 1)

    gmsh.fltk.initialize()
    while gmsh.fltk.is_available():
        gmsh.fltk.wait()
    gmsh.finalize()


def getTriangulation(input: str, triangulationSizing=0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    return nc, inz, C, N, elemTags

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


def __add_as_view_to_gmsh__(tags, data, view_name):
    """
    """
    view = gmsh.view.add(view_name)
    gmsh.view.add_homogeneous_model_data(view, 0, "", "ElementData", tags=tags, data=data, numComponents=1)
    gmsh.view.option.set_number(view, "Visible", 0)
    return view
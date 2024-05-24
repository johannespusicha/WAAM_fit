import gmsh
import os

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

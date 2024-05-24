import gmsh

def show():
    while gmsh.fltk.is_available():
        gmsh.fltk.wait()
    gmsh.finalize()

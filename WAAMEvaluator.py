import gmsh
import numpy as np
from copy import deepcopy
from scipy.optimize import root
import os, sys, random
import tempfile
import platform
import subprocess
from polygonTester import PolygonTester
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

gmsh.initialize()


def evaluateSpheres(input, output, stepSize=0):
    gmsh.model.occ.importShapes(input)
    gmsh.model.occ.synchronize()
    if stepSize != 0:
        gmsh.model.mesh.setSize(gmsh.model.occ.getEntities(0), stepSize)
    gmsh.model.mesh.generate(2)
    nc, inz = __MshFromGmsh__()
    cnts = nc[inz].mean(axis=1)
    r = getSphereRadii(nc, inz)
    gradient = np.zeros_like(r)
    for i in range(gradient.shape[0]):
        neighbours = (np.isin(inz, inz[i]).sum(axis=1) == 2).nonzero()[0]
        gradient[i] = np.linalg.norm((r[i] - r[neighbours]) / np.linalg.norm(cnts[neighbours] - cnts[i], axis=1))
    plotSolid(nc, inz, r, autoLaunch=False)
    print('')


def getSphereRadii(nc, inz):
    faces = gmsh.model.getEntities(2)
    cnts = nc[inz].mean(axis=1)
    faceId = np.zeros(inz.shape[0], dtype=int)
    N = np.zeros(inz.shape)
    for i in range(inz.shape[0]):
        minDst = np.inf
        for face in faces:
            dst = np.linalg.norm(gmsh.model.get_closest_point(2, face[1], cnts[i])[0] - cnts[i])
            if dst < minDst:
                minDst = dst
                faceId[i] = face[1]
        N[i] = -gmsh.model.getNormal(faceId[i], gmsh.model.getParametrization(2, faceId[i], cnts[i]))

    r = -np.ones(inz.shape[0])
    for i in range(inz.shape[0]):
        f = lambda r: __evalRadius__(cnts[i], N[i], faceId[i], r)
        start = 2
        for j in range(10):
            start = start / 2
            solObj = root(f, start)
            if solObj['success']:
                r[i] = solObj['x']
                break
    gmsh.model.remove()
    return r


def plotCurrentGeo():
    fp = os.path.join(tempfile.gettempdir(), '0') + '.step'
    while os.path.exists(fp):
        fp = os.path.join(tempfile.gettempdir(), str(random.randint(0, 10 ** 6))) + '.step'
    gmsh.write(fp)
    subprocess.Popen(
        [sys.executable, os.path.join(os.path.abspath(os.getcwd()), 'GmshPlotter.py'), fp],
        shell=False)


def __evalRadius__(basePnt, N, excludedFace, r):
    faces = gmsh.model.getEntities(2)
    minDst = np.inf
    cnt = basePnt + N * r
    for face in faces:
        if face[1] == excludedFace:
            continue
        x = gmsh.model.get_closest_point(2, face[1], cnt)[0]
        uv = gmsh.model.getParametrization(2, face[1], x)
        if gmsh.model.is_inside(2, face[1], uv, parametric=True) != 0:
            dst = np.linalg.norm(x - cnt)
            if dst < minDst:
                minDst = dst
    return r - minDst


def evaluateIslands(input, output, N=np.array([0, 0, 1]), stepSize=0):
    nc, inz = __getMsh__(input)
    bb = np.reshape(gmsh.model.getBoundingBox(-1, -1), (2, 3))
    cnt = bb.mean(axis=0)
    N = N / np.linalg.norm(N)
    X0 = N * np.inner(cnt - bb[0], N)
    pathLen = np.linalg.norm(bb[1] - bb[0])
    if stepSize <= 0:
        ed = np.vstack([inz[:, [0, 1]], inz[:, [0, 2]], inz[:, [1, 2]]]).T
        stepSize = np.linalg.norm(nc[ed[0]] - nc[ed[1]], axis=1).min()
    steps = int(np.round(pathLen / stepSize))
    stepSize = pathLen/steps
    for i in range(steps):
        X = X0 + N * i * stepSize
        slice = __getSlice(nc, inz, X, N)

    gmsh.model.remove()


def __getSlice(nc, inz, X, N):
    ncOut = np.array([0, 3])
    edOut = np.array([0, 2], dtype=int)

    for i in range(inz.shape[0]):
            trgNodeDir = np.sign(np.inner(X - nc[inz[i]], N))
            if not (trgNodeDir == trgNodeDir[0]).all(): #Is triangle is cut by plane X, N?
                A = nc[inz[i, 0]]
                B = nc[inz[i, 1]]
                C = nc[inz[i, 2]]

                if trgNodeDir[0] == trgNodeDir[1]:
                    print('') #ToDo: complete
                elif trgNodeDir[0] == trgNodeDir[2]:
                    print('') #ToDo: complete

                elif trgNodeDir[1] == trgNodeDir[2]:
                    print('') #ToDo: complete
    print('')


def __getMsh__(input):
    if input[-3:] == 'stp' or input[-4:] == 'step':
        gmsh.model.occ.importShapes(input)
        gmsh.model.occ.synchronize()
    elif input[-3:] == 'stl':
        gmsh.open(input)
    else:
        raise 'unknown format'
    return __MshFromGmsh__()


def __MshFromGmsh__():
    nodeCoords = np.array([], dtype=float)
    gmsh.model.mesh.renumber_nodes()
    for i in range(0, 3):
        rnodeCoords = gmsh.model.mesh.getNodes(i)[1]
        nodeCoords = np.append(nodeCoords, rnodeCoords)
    nc = nodeCoords.reshape(-1, 3)
    elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(2)
    for i in range(len(elemTypes)):
        elemTypes[i] = elemTypes[i].astype(int)
        elemTags[i] = elemTags[i].astype(int)
        elemNodeTags[i] = elemNodeTags[i].astype(int).reshape(elemTags[i].shape[0], -1) - 1
    if len(elemTypes) == 0:
        return np.zeros((0, 3)), np.zeros(0, dtype=int)
    else:
        return nc, elemNodeTags[0]


def __exportToOpenSCAD__(msh, outPath, elemNames=None, colors=None):
    elemTypeShortNames = {2: 'Triangle', 3: 'Quad', 4: 'Tetrahedron', 5: 'Hexahedron', 6: 'Prism', 7: 'Pyramid'}
    elemColors = {2: 'red', 3: 'blue', 4: 'blue', 5: 'red', 6: 'green', 7: 'yellow'}

    fileContent = ['//Mesh exported using MeshTools.exportToOpenSCAD\n']

    nc = msh['nc']

    # write point coordinates
    fileContent += ['Points = [\n']
    for i in range(nc.shape[0]):
        fileContent += [str(nc[i].tolist()) + ', // ' + str(i) + '\n']
    fileContent += ['  ];\n']

    # write elements
    e = 0
    for etIndex in range(msh['elemTypes'].shape[0]):
        et = msh['elemTypes'][etIndex]
        etFaces = [np.array([0, 1, 2], dtype=int)]
        inz = msh['inz'][etIndex]
        for i in range(inz.shape[0]):
            if elemNames is None:
                elemName = elemTypeShortNames[et] + str(i)
            else:
                elemName = str(elemNames[et][i])
            fileContent += ['\n' + elemName + ' = [\n']
            cnt = msh['nc'][inz[0]].mean(axis=0)
            for face in etFaces:
                N = np.cross(nc[inz[i, face[1]]] - nc[inz[i, face[0]]], nc[inz[i, face[2]]] - nc[inz[i, face[1]]])
                if np.inner(np.mean(nc[inz[i, face]], axis=0) - cnt, N) > 0:
                    fileContent += [str(inz[i, face].tolist()) + ',']
                else:
                    fileContent += [str(inz[i, np.flip(face)].tolist()) + ',']
            fileContent[-1] = fileContent[-1][:-1]
            fileContent += ['  ];\n']

            fileContent += ['//[' + str(inz[i].tolist()) + ']\n']
            if colors is None:
                fileContent += ['color("' + elemColors[et] + '") polyhedron( Points, ' + elemName + ' );\n']
            else:
                fileContent += ['color(' + colors[e] + ') polyhedron( Points, ' + elemName + ' );\n']
            e = e + 1

    fileContent += ['LineWidth = 0.03;\n']

    fileContent += ['module line(start, end, thickness) {\n']
    fileContent += ['    color("black") hull() {\n']
    fileContent += ['        translate(start) sphere(thickness);\n']
    fileContent += ['        translate(end) sphere(thickness);\n']
    fileContent += ['    }\n']
    fileContent += ['}\n']

    if not outPath[-5:] == '.scad':
        outPath += '.scad'
    fp = open(outPath, 'w')
    fp.writelines(fileContent)
    fp.close()


def plotSolid(nc, inz, value, autoLaunch=True):
    scadPath = os.path.join(tempfile.gettempdir(), 'out.scad')
    value = value - value.min()
    value = value/value.max()
    clormap = [''] * inz.shape[0]
    for i in range(inz.shape[0]):
        clormap[i] = '[' + str(value[i]) + ', 0., ' + str(1 - value[i]) + ']'

    __exportToOpenSCAD__({'nc': nc, 'inz': [inz], 'elemTypes': np.array([2])}, scadPath, colors=clormap)
    if autoLaunch:
        if platform.system() == 'Darwin':
            subprocess.call(('open', scadPath))
        elif platform.system() == 'Windows':
            os.startfile(scadPath)
        else:
            subprocess.call(('xdg-open', scadPath))

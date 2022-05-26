import gmsh
import numpy as np
from scipy.optimize import root
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

    r = np.zeros(inz.shape[0])
    for i in range(inz.shape[0]):
        f = lambda r: __evalRadius__(cnts[i], N[i], faceId[i], r)
        r[i] = root(f, 0)['x']
    gmsh.model.remove()
    return r

def __evalRadius__(basePnt, N, excludedFace,r):
    faces = gmsh.model.getEntities(2)
    minDst = np.inf
    cnt = basePnt + N * r
    for face in faces:
        if face[1] == excludedFace:
            continue
        dst = np.linalg.norm(gmsh.model.get_closest_point(2, face[1], cnt)[0] - cnt)
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

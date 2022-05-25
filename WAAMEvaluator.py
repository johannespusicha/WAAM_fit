import gmsh
import numpy as np
from polygonTester import PolygonTester
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

gmsh.initialize()


def __step2gmsh__(geoFilePath):
    gmsh.model.remove()
    gmsh.model.add('model')
    gmsh.model.occ.importShapes(geoFilePath)
    gmsh.model.occ.synchronize()


def evaluate(input, output, strides=np.array([10, 10, 10])):
    __step2gmsh__(input)
    pointInPolygonTester = PolygonTester()
    boundingBox = np.array(gmsh.model.get_bounding_box(-1, -1)).reshape([-1, 3])
    dx = (boundingBox[1] - boundingBox[0]) / strides
    x, y, z = np.meshgrid(np.linspace(boundingBox[0, 0], boundingBox[1, 0], strides[0]),
                          np.linspace(boundingBox[0, 1], boundingBox[1, 1], strides[1]),
                          np.linspace(boundingBox[0, 2], boundingBox[1, 2], strides[2]))
    pnts = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T
    r = __getRadii__(pnts).reshape(strides)
    grad = np.zeros([strides[0] - 2, strides[1] - 2, strides[2] - 2])
    inInPnts = pointInPolygonTester.pntInPolygon(pnts).reshape(strides.tolist())
    isIn = inInPnts[1:-1, 1:-1, 1:-1].flatten()

    for i in range(1, strides[0] - 1):
        for j in range(1, strides[1] - 1):
            for k in range(1, strides[2] - 1):
                grad[i - 1, j - 1, k - 1] = np.linalg.norm([(r[i + 1, j, k] - r[i - 1, j, k]) / dx[0],
                                                            (r[i, j + 1, k] - r[i, j - 1, k]) / dx[1],
                                                            (r[i, j, k + 1] - r[i, j, k - 1]) / dx[2]])


    sgrad = ((grad - grad.min()) / (grad.max() - grad.min())).flatten()

    plotPnts = np.vstack([x[1:-1, 1:-1, 1:-1].flatten(), y[1:-1, 1:-1, 1:-1].flatten(), z[1:-1, 1:-1, 1:-1].flatten()]).T

    plotColor = np.vstack([sgrad, np.zeros_like(sgrad), 1 - sgrad]).T
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    isIn[np.logical_not(np.logical_and(0 < plotPnts[:, 2], plotPnts[:, 2] < 0.5))] = False
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.scatter(plotPnts[isIn, 0], plotPnts[isIn, 1], plotPnts[isIn, 2], c=plotColor[isIn])

    gmsh.model.remove()


def __getRadii__(pnts):
    r = np.full(pnts.shape[0], np.inf)
    faces = gmsh.model.getEntities(2)
    for i in range(r.shape[0]):
        for face in faces:
            S = gmsh.model.get_closest_point(2, face[1], pnts[i])[0]
            r[i] = min(r[i], np.linalg.norm(S - pnts[i]))
    return r
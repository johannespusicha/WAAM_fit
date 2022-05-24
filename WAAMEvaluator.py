import gmsh
import numpy as np
from scipy.spatial import cKDTree

gmsh.initialize()


def __step2gmsh__(geoFilePath):
    gmsh.model.remove()
    gmsh.model.add('model')
    gmsh.model.occ.importShapes(geoFilePath)
    gmsh.model.occ.synchronize()


def evaluate(input, output):
    ev = Evaluator(input)
    ev.GetGeoProperties()
    gmsh.model.remove()


class Evaluator:
    def __init__(self, input):
        __step2gmsh__(input)
        gmsh.model.mesh.generate(2)
        #ToDo: maybe apply sizing

        self.HullPnts = np.zeros([0, 3])
        self.HullEntityIDs = np.zeros(0, dtype=int)
        self.HullEntityDims = np.zeros(0, dtype=int)
        self.HullParameters = np.zeros([0, 2])
        self.PntOutNormals = np.zeros([0, 3])

        acc = []
        for item in gmsh.model.getEntities(0):
            acc += [item[1]]
        self.corners = np.array(acc, dtype=int)
        acc = []
        for item in gmsh.model.getEntities(1):
            acc += [item[1]]
        self.curves = np.array(acc, dtype=int)
        acc = []
        for item in gmsh.model.getEntities(2):
            acc += [item[1]]
        self.faces = np.array(acc, dtype=int)

        self.curveBnds = []
        for curve in self.curves:
            rawBnd = gmsh.model.getBoundary([[1, curve]], combined=False)
            self.curveBnds += [[]]
            for bnd in rawBnd:
                self.curveBnds[-1] += [abs(bnd[1])]

        self.faceBdIndicis = []
        for face in self.faces:
            rawBnd = gmsh.model.getBoundary([[2, face]], combined=False)
            self.faceBdIndicis += [[]]
            for bnd in rawBnd:
                self.faceBdIndicis[-1] += [(self.curves == abs(bnd[1])).nonzero()[0][0]]

        for i in range(self.corners.shape[0]):
            tag = self.corners[i]
            EntityNodes = gmsh.model.mesh.get_nodes(0, tag)[1].reshape([-1, 3])
            self.HullPnts = np.vstack([self.HullPnts, EntityNodes])
            self.HullEntityDims = np.concatenate([self.HullEntityDims, np.ones(EntityNodes.shape[0], dtype=int) * 0])
            self.HullEntityIDs = np.concatenate([self.HullEntityIDs, np.ones(EntityNodes.shape[0], dtype=int) * tag])
            self.HullParameters = np.vstack([self.HullParameters, np.zeros([EntityNodes.shape[0], 2])])
            AdjacentNormals = np.zeros([0, 3])
            CrvsAtCorner = (self.curveBnds == tag).any(axis=1).nonzero()[0]
            for j in CrvsAtCorner:
                for k in range(self.faces.shape[0]):
                    if j in self.faceBdIndicis[k]:
                        faceParameters = gmsh.model.getParametrization(2, self.faces[k], EntityNodes[0])
                        AdjacentNormals = np.vstack([AdjacentNormals, gmsh.model.getNormal(self.faces[k], faceParameters)])
            self.PntOutNormals = np.vstack([self.PntOutNormals, AdjacentNormals.mean(axis=0)])

        for i in range(self.curves.shape[0]):
            tag = self.curves[i]
            EntityNodes = gmsh.model.mesh.get_nodes(1, tag)[1].reshape([-1, 3])
            self.HullPnts = np.vstack([self.HullPnts, EntityNodes])
            self.HullEntityDims = np.concatenate([self.HullEntityDims, np.ones(EntityNodes.shape[0], dtype=int) * 1])
            self.HullEntityIDs = np.concatenate([self.HullEntityIDs, np.ones(EntityNodes.shape[0], dtype=int) * tag])
            RawParameters = gmsh.model.mesh.get_nodes(1, tag)[2]
            self.HullParameters = np.vstack([self.HullParameters, np.vstack([RawParameters, np.zeros_like(RawParameters)]).T])
            for j in range(EntityNodes.shape[0]):
                AdjacentNormals = np.zeros([0, 3])
                for k in range(self.faces.shape[0]):
                    if i in self.faceBdIndicis[k]:
                        faceParameters = gmsh.model.getParametrization(2, self.faces[k], EntityNodes[0])
                        AdjacentNormals = np.vstack([AdjacentNormals, gmsh.model.getNormal(self.faces[k], faceParameters)])
                self.PntOutNormals = np.vstack([self.PntOutNormals, AdjacentNormals.mean(axis=0)])

        for i in range(self.faces.shape[0]):
            tag = self.faces[i]
            EntityNodes = gmsh.model.mesh.get_nodes(2, tag)[1].reshape([-1, 3])
            self.HullPnts = np.vstack([self.HullPnts, EntityNodes])
            self.HullEntityDims = np.concatenate([self.HullEntityDims, np.ones(EntityNodes.shape[0], dtype=int) * 2])
            self.HullEntityIDs = np.concatenate([self.HullEntityIDs, np.ones(EntityNodes.shape[0], dtype=int) * tag])
            RawParameters = gmsh.model.mesh.get_nodes(2, tag)[2]
            self.HullParameters = np.vstack([self.HullParameters, RawParameters.reshape([-1, 2])])
            self.PntOutNormals = np.vstack([self.PntOutNormals, gmsh.model.get_normal(tag, RawParameters).reshape([-1, 3])])

        self.PntTree = cKDTree(self.HullPnts)
        self.inz = gmsh.model.mesh.getElements(2)[2][0].reshape([-1, 3]) - 1
        self.N = np.cross(self.HullPnts[self.inz[:, 0]] - self.HullPnts[self.inz[:, 1]], self.HullPnts[self.inz[:, 0]] - self.HullPnts[self.inz[:, 2]])
        self.N = self.N / np.tile(np.linalg.norm(self.N, axis=1), (3, 1)).T
        self.N = self.N * np.tile(np.sign((self.N * self.PntOutNormals[self.inz[:, 0]]).sum(axis=1)), (3, 1)).T
        self.trgCnt = self.HullPnts[self.inz].mean(axis=1)

    def __del__(self):
        gmsh.model.remove()

    def GetGeoProperties(self):
        self.WallTickness = self.__getWallThickness__()

    def __getWallThickness__(self):
        res = -np.ones([self.inz.shape[0]])
        for i in range(self.inz.shape[0]):
            for j in range(self.inz.shape[0]):
                if np.sign(np.inner(self.N[i], self.N[j])) >= 0:
                    continue # normal has false orientation
                N = self.N[i]
                A = self.HullPnts[self.inz[j, 0]]
                B = self.HullPnts[self.inz[j, 1]]
                C = self.HullPnts[self.inz[j, 2]]
                M = self.trgCnt[i]
                EqRes = np.linalg.solve(np.stack([N, B-A, C-A]).T, M-A)
                if (EqRes >= 0).all() and (EqRes[1:] <= 1).all() and EqRes[1] <= 1 - EqRes[2]:
                    res[i] = EqRes[0]
        print('')

def __evalRadius__(ncTree, inz, baseTrg, uv, r):


    print('')
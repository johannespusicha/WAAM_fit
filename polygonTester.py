import numpy as np
from scipy.spatial import cKDTree
from numba import njit, int32, int64
import gmsh


def getUniqueLines(inmat):
    mat = inmat.copy()
    mat.sort(axis=1)
    mat = np.flip(mat, axis=1)
    lexOrder = np.lexsort(mat.T)
    count = __getUniqueLinesInternal__(mat, lexOrder)
    return lexOrder[count > 0], count[count > 0]


def __MshFromGmsh__(dim=3):
    ElemOrder = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 0,
                 16: 2, 17: 2, 18: 2, 19: 2, 20: 3, 21: 3, 22: 4, 23: 4, 24: 5, 25: 5, 26: 3, 27: 4,
                 28: 5, 29: 3, 30: 4, 31: 5}
    nodeCoords = np.array([], dtype=float)
    gmsh.model.mesh.renumber_nodes()
    for i in range(0, dim + 1):
        rnodeCoords = gmsh.model.mesh.getNodes(i)[1]
        nodeCoords = np.append(nodeCoords, rnodeCoords)
    nc = nodeCoords.reshape(-1, 3)
    elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim)
    for i in range(len(elemTypes)):
        elemTypes[i] = elemTypes[i].astype(int)
        elemTags[i] = elemTags[i].astype(int)
        elemNodeTags[i] = elemNodeTags[i].astype(int).reshape(elemTags[i].shape[0], -1) - 1
    if len(elemTypes) == 0:
        Msh = {'nc': np.zeros((0, 3)), 'elemTypes': np.zeros(0, dtype=int),
                'inz': [], 'Order': 1, 'ForceMidplaneNodes': False}
    else:
        Msh = {'nc': nc, 'elemTypes': elemTypes,
                'inz': elemNodeTags, 'Order': ElemOrder[elemTypes[0]], 'ForceMidplaneNodes': False}

    return Msh


@njit(int32[:](int32[:, :], int64[:]), cache=True)
def __getUniqueLinesInternal__(mat, lexOrder):
    count = np.zeros(mat.shape[0], dtype=int32)
    countedIndex = 0
    count[0] = 1
    countedValue = mat[lexOrder[countedIndex]]
    for i in range(1, mat.shape[0]):
        if not (mat[lexOrder[i]] == countedValue).all():
            countedIndex = i
            countedValue = mat[lexOrder[countedIndex]]
        count[countedIndex] += 1
    return count


@njit(int32[:, :](int32[:, :], int32[:, :], int32), cache=True)
def __getTrgWithSharedEdgeInternal__(inz, allEd, maxNodeID):
    trgWithSharedEdge = np.zeros_like(allEd)
    trgAtNodes = [[0]]
    for i in range(maxNodeID + 1):
        trgAtNodes.append([-1])
    for i in range(inz.shape[0]):
        trgAtNodes[inz[i, 0]] += [i]
        trgAtNodes[inz[i, 1]] += [i]
        trgAtNodes[inz[i, 2]] += [i]

    for i in range(allEd.shape[0]):
        trgWithSharedEdge[i] = np.intersect1d(trgAtNodes[allEd[i, 0]][1:], trgAtNodes[allEd[i, 1]][1:])
    return trgWithSharedEdge


def __getTrgWithSharedEdge__(inz):
    allEd = np.vstack([inz[:, [0, 1]], inz[:, [0, 2]], inz[:, [1, 2]]])
    allEd = allEd[getUniqueLines(allEd)[0]]
    return __getTrgWithSharedEdgeInternal__(inz, allEd, inz.max())


class PolygonTester:
    def __init__(self, coord=None, inz=None):
        if coord is None:
            gmsh.model.mesh.generate(2)
            msh = __MshFromGmsh__(2)
            coord = msh['nc']
            inz = msh['inz'][0]
        self.coord = coord
        self.inz = inz

        self.trgWithSharedEdge = __getTrgWithSharedEdge__(inz)

        TrgCnts = coord[inz].mean(axis=1)
        self.projTrgCnts = coord[inz, 1:].mean(axis=1)
        self.projTrgRadii = np.stack([np.linalg.norm(coord[inz[:, 0]] - TrgCnts, axis=1),
                                      np.linalg.norm(coord[inz[:, 1]] - TrgCnts, axis=1),
                                      np.linalg.norm(coord[inz[:, 2]] - TrgCnts, axis=1)]).max(axis=0)
        self.trgCntTree = cKDTree(self.projTrgCnts)
        self.trgCnt3D = coord[inz].mean(axis=1)

        self.N = np.cross(coord[inz[:, 1]] - coord[inz[:, 0]], coord[inz[:, 2]] - coord[inz[:, 0]])
        self.N = self.N / np.tile(np.linalg.norm(self.N, axis=1), (3, 1)).T

        self.S = np.mean(self.coord[self.inz], axis=1)

        self.trgCntTree3D = cKDTree(self.trgCnt3D)
        self.maxRadius = self.projTrgRadii.max()
        self.trgCount = inz.shape[0]

        self.planes = [[0, 1], [0, 2], [1, 2]]
        self.mes = []
        self.vis = []
        for plane in self.planes:
            A = coord[inz[:, 0]][:, plane]
            B = coord[inz[:, 1]][:, plane]
            C = coord[inz[:, 2]][:, plane]
            vi = np.zeros([inz.shape[0], 3, 2])
            me = np.zeros([inz.shape[0], 3, 2])
            for i in range(inz.shape[0]):
                me[i, 0] = np.mean([B[i], C[i]], axis=0)
                me[i, 1] = np.mean([A[i], C[i]], axis=0)
                me[i, 2] = np.mean([B[i], A[i]], axis=0)
                vi[i, 0] = np.matmul(np.array([[0, -1], [1, 0]]), C[i] - B[i])
                vi[i, 0] *= np.sign(np.inner(vi[i, 0], A[i] - me[i, 0]))
                vi[i, 1] = np.matmul(np.array([[0, -1], [1, 0]]), A[i] - C[i])
                vi[i, 1] *= np.sign(np.inner(vi[i, 1], B[i] - me[i, 1]))
                vi[i, 2] = np.matmul(np.array([[0, -1], [1, 0]]), A[i] - B[i])
                vi[i, 2] *= np.sign(np.inner(vi[i, 2], C[i] - me[i, 2]))
            self.mes += [me]
            self.vis += [vi]

    def pntInPolygon(self, pnts, returnOutDists=False):
        res = np.full(pnts.shape[0], True, dtype=bool)
        for i in range(pnts.shape[0]):
            pnt = pnts[i]
            pntIsIn = np.zeros(3, dtype=bool)

            noneDegeneratedTrgs = np.abs(self.N[:, 2]) > 10 ** -1
            trgsCollidingWithPntLocalIndexing = self.__pntInTrgs__(pnt[[0, 1]], 0, noneDegeneratedTrgs)
            trgsCollidingWithPnt = noneDegeneratedTrgs.nonzero()[0][trgsCollidingWithPntLocalIndexing]
            pntIsIn[0] = ((self.S[trgsCollidingWithPnt] - pnt)[:, 2] >= 0).sum() % 2 != 0

            noneDegeneratedTrgs = np.abs(self.N[:, 1]) > 10 ** -10
            trgsCollidingWithPntLocalIndexing = self.__pntInTrgs__(pnt[[0, 2]], 1, noneDegeneratedTrgs)
            trgsCollidingWithPnt = noneDegeneratedTrgs.nonzero()[0][trgsCollidingWithPntLocalIndexing]
            pntIsIn[1] = ((self.S[trgsCollidingWithPnt] - pnt)[:, 1] >= 0).sum() % 2 != 0

            noneDegeneratedTrgs = np.abs(self.N[:, 0]) > 10 ** -10
            trgsCollidingWithPntLocalIndexing = self.__pntInTrgs__(pnt[[1, 2]], 2, noneDegeneratedTrgs)
            trgsCollidingWithPnt = noneDegeneratedTrgs.nonzero()[0][trgsCollidingWithPntLocalIndexing]
            pntIsIn[2] = ((self.S[trgsCollidingWithPnt] - pnt)[:, 0] >= 0).sum() % 2 != 0

            res[i] = pntIsIn.all()

        if returnOutDists:
            srfDst = np.zeros(pnts.shape[0])
            for i in range(pnts.shape[0]):
                treeRes = self.trgCntTree3D.query(pnts[i])
                if np.isinf(treeRes[0]):
                    srfDst[i] = np.inf
                else:
                    srfDst[i] = abs(np.inner(self.trgCntTree3D.data[treeRes[1]] - pnts[i], self.N[treeRes[1]]))
            return res, srfDst
        else:
            return res

    def __pntInTrgs__(self, P, plane, noneDegeneratedTrgs):
        vi = self.vis[plane][noneDegeneratedTrgs]
        me = self.mes[plane][noneDegeneratedTrgs]
        globalIndicisOfTrgs = np.arange(self.inz.shape[0])[noneDegeneratedTrgs]
        localIndicisOfGlobalTrg = noneDegeneratedTrgs.nonzero()[0]

        colTrgs = np.stack([np.sign(np.sum((P - me[:, 0]) * vi[:, 0], axis=1)) > 0,
                            np.sign(np.sum((P - me[:, 1]) * vi[:, 1], axis=1)) > 0,
                            np.sign(np.sum((P - me[:, 2]) * vi[:, 2], axis=1)) > 0]).all(axis=0)

        nodesHit = (self.coord[:, self.planes[plane]] == P).all(axis=1).nonzero()[0]
        trgForHitNode = []
        for i in nodesHit:
            allTrgAtNode = np.isin(self.inz[noneDegeneratedTrgs], i).any(axis=1).nonzero()[0]
            if allTrgAtNode.shape[0] > 0:
                trgForHitNode += [allTrgAtNode[0]]

        trgWithEdgeHit = np.stack([np.sign(np.sum((P - me[:, 0]) * vi[:, 0], axis=1)) == 0,
                                   np.sign(np.sum((P - me[:, 1]) * vi[:, 1], axis=1)) > 0,
                                   np.sign(np.sum((P - me[:, 2]) * vi[:, 2], axis=1)) > 0]).all(axis=0)

        trgWithEdgeHit = np.logical_or(trgWithEdgeHit,
                  np.stack([np.sign(np.sum((P - me[:, 0]) * vi[:, 0], axis=1)) > 0,
                            np.sign(np.sum((P - me[:, 1]) * vi[:, 1], axis=1)) == 0,
                            np.sign(np.sum((P - me[:, 2]) * vi[:, 2], axis=1)) > 0]).all(axis=0))

        trgWithEdgeHit = np.logical_or(trgWithEdgeHit,
                  np.stack([np.sign(np.sum((P - me[:, 0]) * vi[:, 0], axis=1)) > 0,
                            np.sign(np.sum((P - me[:, 1]) * vi[:, 1], axis=1)) > 0,
                            np.sign(np.sum((P - me[:, 2]) * vi[:, 2], axis=1)) == 0]).all(axis=0))

        globalFirstTrgOfEdgeHit = self.trgWithSharedEdge[np.isin(self.trgWithSharedEdge, globalIndicisOfTrgs[trgWithEdgeHit]).all(axis=1), 0]

        return np.concatenate([colTrgs.nonzero()[0], np.isin(localIndicisOfGlobalTrg, globalFirstTrgOfEdgeHit).nonzero()[0], np.array(trgForHitNode, dtype=int)])

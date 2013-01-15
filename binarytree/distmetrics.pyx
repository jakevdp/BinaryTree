#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, pow, fabs

DTYPE = np.float64
#ctypedef np.float64_t DTYPE_t

# warning: there will be problems if ITYPE
#  is switched to an unsigned type!
ITYPE = np.intp
#ctypedef np.intp_t ITYPE_t


cdef class _DistanceMetric:
    cdef DTYPE_t dist(self, DTYPE_t[:, ::1] X1, ITYPE_t i1,
                      DTYPE_t[:, ::1] X2, ITYPE_t i2):
        return 0.0

    def pairwise(self, X, Y=None):
        """Compute pairwise distances between arrays
        
        Parameters
        ----------
        X : array_like
            size `(nX, d)` matrix of `nX` points in `d` dimensions.
        Y : array_like (optional)
            size `(nY, d)` matrix of `nY` points in `d` dimensions.
            If not specified, then Y = X and the computation will take
            into account the symmetry.

        Returns
        -------
        D : array_like
            size `(nX, nY)` matrix of distances, where `D[i, j]` is the distance
            between point `X[i]` and `Y[j]`.
        """
        X = np.asarray(X)
        if Y is None:
            D = self.pdist(X)
        else:
            Y = np.asarray(Y)
            D = self.cdist(X, Y)
        return np.asarray(D)        

    cdef DTYPE_t[:, ::1] pdist(self, DTYPE_t[:, ::1] X):
        cdef ITYPE_t nX = X.shape[0]
        cdef DTYPE_t[:, ::1] D = np.zeros((nX, nX),
                                          dtype=DTYPE, order='C')
        for i1 in range(nX):
            for i2 in range(i1, nX):
                D[i1, i2] = self.dist(X, i1, X, i2)
                D[i2, i1] = D[i1, i2]
        return D

    cdef DTYPE_t[:, ::1] cdist(self, DTYPE_t[:, ::1] X, DTYPE_t[:, ::1] Y):
        cdef ITYPE_t nX = X.shape[0]
        cdef ITYPE_t nY = Y.shape[0]
        if X.shape[1] != Y.shape[1]:
            raise ValueError('X and Y must have the same second dimension')

        cdef DTYPE_t[:, ::1] D = np.zeros((nX, nX),
                                            dtype=DTYPE, order='C')
        for i1 in range(nX):
            for i2 in range(nY):
                D[i1, i2] = self.dist(X, i1, Y, i2)
        return D


cdef class EuclideanDistance(_DistanceMetric):
    cdef DTYPE_t dist(self, DTYPE_t[:, ::1] X1, ITYPE_t i1,
                      DTYPE_t[:, ::1] X2, ITYPE_t i2):
        cdef ITYPE_t n_features = X1.shape[1]
        cdef DTYPE_t tmp, d=0
        for j in range(n_features):
            tmp = X1[i1, j] - X2[i2, j]
            d += tmp * tmp
        return sqrt(d)


cdef class ManhattanDistance(_DistanceMetric):
    cdef DTYPE_t dist(self, DTYPE_t[:, ::1] X1, ITYPE_t i1,
                      DTYPE_t[:, ::1] X2, ITYPE_t i2):
        cdef ITYPE_t n_features = X1.shape[1]
        cdef DTYPE_t tmp, d=0
        for j in range(n_features):
            d += fabs(X1[i1, j] - X2[i2, j])
        return d


cdef class MinkowskiDistance(_DistanceMetric):
    def __init__(self, p=2):
        if p <= 0:
            raise ValueError("p must be positive")
        self.p = p

    @cython.cdivision(True)
    cdef DTYPE_t dist(self, DTYPE_t[:, ::1] X1, ITYPE_t i1,
                      DTYPE_t[:, ::1] X2, ITYPE_t i2):
        cdef ITYPE_t n_features = X1.shape[1]
        cdef DTYPE_t tmp, d=0
        for j in range(n_features):
            d += pow(fabs(X1[i1, j] - X2[i2, j]), self.p)
        return pow(d, 1. / self.p)


def DistanceMetric(metric, **kwargs):
    if metric in ['euclidean', 'l2']:
        return EuclideanDistance(**kwargs)
    elif metric in ['manhattan', 'l1']:
        return ManhattanDistance(**kwargs)
    elif metric == 'minkowski':
        return MinkowskiDistance(**kwargs)
    else:
        raise ValueError("Unrecognized metric '%s'" % str(metric))
    

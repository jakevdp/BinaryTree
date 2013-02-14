#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
cimport cython
cimport numpy as np
from libc.math cimport fmax, fmin, fabs, sqrt

import numpy as np
from typedefs import DTYPE, ITYPE
from typedefs cimport DTYPE_t, ITYPE_t


############################################################
# Euclidean Inline Functions
cdef inline DTYPE_t euclidean_dist(DTYPE_t[:, ::1] X1, ITYPE_t i1,
                                   DTYPE_t[:, ::1] X2, ITYPE_t i2):
    cdef DTYPE_t tmp, d=0
    for j in range(X1.shape[1]):
        tmp = X1[i1, j] - X2[i2, j]
        d += tmp * tmp
    return sqrt(d)

cdef DTYPE_t[:, ::1] euclidean_cdist(DTYPE_t[:, ::1] X, DTYPE_t[:, ::1] Y):
    if X.shape[1] != Y.shape[1]:
        raise ValueError('X and Y must have the same second dimension')

    cdef DTYPE_t[:, ::1] D = np.zeros((X.shape[0], Y.shape[0]), dtype=DTYPE)

    for i1 in range(X.shape[0]):
        for i2 in range(Y.shape[0]):
            D[i1, i2] = euclidean_dist(X, i1, Y, i2)
    return D


############################################################
# Base class for distance metrics
cdef class DistanceMetric:
    def __init__(self, **kwargs):
        if self.__class__ is DistanceMetric:
            raise NotImplementedError("DistanceMetric is an abstract class")

    cdef DTYPE_t dist(self, DTYPE_t[:, ::1] X1, ITYPE_t i1,
                      DTYPE_t[:, ::1] X2, ITYPE_t i2):
        return -999

    cdef DTYPE_t rdist(self, DTYPE_t[:, ::1] X1, ITYPE_t i1,
                       DTYPE_t[:, ::1] X2, ITYPE_t i2):
        return self.dist(X1, i1, X2, i2)

    def rdist_to_dist(self, rdist):
        return rdist

    def dist_to_rdist(self, dist):
        return dist

    def pairwise(self, X, Y=None):
        X = np.asarray(X, dtype=DTYPE)
        if Y is None:
            D = self.pdist(X)
        else:
            Y = np.asarray(Y, dtype=DTYPE)
            D = self.cdist(X, Y)
        return np.asarray(D)

    cdef DTYPE_t[:, ::1] pdist(self, DTYPE_t[:, ::1] X):
        cdef ITYPE_t i1, i2
        cdef DTYPE_t[:, ::1] D = np.zeros((X.shape[0], X.shape[0]),
                                          dtype=DTYPE, order='C')
        for i1 in range(X.shape[0]):
            for i2 in range(i1, X.shape[0]):
                D[i1, i2] = self.dist(X, i1, X, i2)
                D[i2, i1] = D[i1, i2]
        return D

    cdef DTYPE_t[:, ::1] cdist(self, DTYPE_t[:, ::1] X, DTYPE_t[:, ::1] Y):
        cdef ITYPE_t i1, i2
        if X.shape[1] != Y.shape[1]:
            raise ValueError('X and Y must have the same second dimension')

        cdef DTYPE_t[:, ::1] D = np.zeros((X.shape[0], Y.shape[0]),
                                          dtype=DTYPE, order='C')

        for i1 in range(X.shape[0]):
            for i2 in range(Y.shape[0]):
                D[i1, i2] = self.dist(X, i1, Y, i2)
        return D


############################################################
# Specializations of distance metrics
cdef class EuclideanDistance(DistanceMetric):
    def __init__(self):
        pass

    cdef inline DTYPE_t dist(self, DTYPE_t[:, ::1] X1, ITYPE_t i1,
                             DTYPE_t[:, ::1] X2, ITYPE_t i2):
        cdef DTYPE_t tmp, d=0
        for j in range(X1.shape[1]):
            tmp = X1[i1, j] - X2[i2, j]
            d += tmp * tmp
        return sqrt(d)

    cdef inline DTYPE_t rdist(self, DTYPE_t[:, ::1] X1, ITYPE_t i1,
                              DTYPE_t[:, ::1] X2, ITYPE_t i2):
        cdef DTYPE_t tmp, d=0
        for j in range(X1.shape[1]):
            tmp = X1[i1, j] - X2[i2, j]
            d += tmp * tmp
        return d

    def rdist_to_dist(self, rdist):
        return np.sqrt(rdist)

    def dist_to_rdist(self, dist):
        return dist ** 2


############################################################
# python access functions, used for benchmarking

def euclidean_pairwise(DTYPE_t[:, ::1] X, DTYPE_t[:, ::1] Y):
    D = euclidean_cdist(X, Y)
    return np.asarray(D)


def euclidean_pairwise_class(DTYPE_t[:, ::1] X, DTYPE_t[:, ::1] Y):
    cdef EuclideanDistance eucl_dist = EuclideanDistance()

    assert X.shape[1] == Y.shape[1]
    cdef DTYPE_t[:, ::1] D = np.zeros((X.shape[0], Y.shape[0]), dtype=DTYPE)

    for i1 in range(X.shape[0]):
        for i2 in range(Y.shape[0]):
            D[i1, i2] = eucl_dist.dist(X, i1, Y, i2)
    return np.asarray(D)


def euclidean_pairwise_polymorph(DTYPE_t[:, ::1] X, DTYPE_t[:, ::1] Y):
    cdef DistanceMetric eucl_dist = EuclideanDistance()

    assert X.shape[1] == Y.shape[1]
    cdef DTYPE_t[:, ::1] D = np.zeros((X.shape[0], Y.shape[0]), dtype=DTYPE)

    for i1 in range(X.shape[0]):
        for i2 in range(Y.shape[0]):
            D[i1, i2] = eucl_dist.dist(X, i1, Y, i2)
    return np.asarray(D)



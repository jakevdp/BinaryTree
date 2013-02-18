#!python
from typedefs cimport DTYPE_t, ITYPE_t
from libc.math cimport fmax, fmin, fabs, sqrt

############################################################
# Euclidean Inline Functions
cdef inline DTYPE_t euclidean_dist(DTYPE_t[:, ::1] X1, ITYPE_t i1,
                                   DTYPE_t[:, ::1] X2, ITYPE_t i2):
    cdef DTYPE_t tmp, d=0
    for j in range(X1.shape[1]):
        tmp = X1[i1, j] - X2[i2, j]
        d += tmp * tmp
    return sqrt(d)


cdef class DistanceMetric:
    cdef DTYPE_t dist(self, DTYPE_t[:, ::1] X1, ITYPE_t i1,
                      DTYPE_t[:, ::1] X2, ITYPE_t i2)

    cdef DTYPE_t rdist(self, DTYPE_t[:, ::1] X1, ITYPE_t i1,
                       DTYPE_t[:, ::1] X2, ITYPE_t i2)

    cdef DTYPE_t[:, ::1] pdist(self, DTYPE_t[:, ::1] X)

    cdef DTYPE_t[:, ::1] cdist(self, DTYPE_t[:, ::1] X, DTYPE_t[:, ::1] Y)


cdef class EuclideanDistance(DistanceMetric):
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

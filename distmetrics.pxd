#!python
import numpy as np
cimport numpy as np
cimport cython

from typedefs cimport DTYPE_t, ITYPE_t


cdef class DistanceMetric:
    cdef int axis_aligned
    cdef DTYPE_t p

    cdef DTYPE_t dist(self, DTYPE_t[:, ::1] X1, ITYPE_t i1,
                      DTYPE_t[:, ::1] X2, ITYPE_t i2)
    cdef DTYPE_t rdist(self, DTYPE_t[:, ::1] X1, ITYPE_t i1,
                      DTYPE_t[:, ::1] X2, ITYPE_t i2)
    cdef DTYPE_t[:, ::1] pdist(self, DTYPE_t[:, ::1])
    cdef DTYPE_t[:, ::1] cdist(self, DTYPE_t[:, ::1], DTYPE_t[:, ::1])


@cython.final
cdef class EuclideanDistance(DistanceMetric):
    pass


@cython.final
cdef class ManhattanDistance(DistanceMetric):
    pass


@cython.final
cdef class MinkowskiDistance(DistanceMetric):
    pass

@cython.final
cdef class ChebyshevDistance(DistanceMetric):
    pass

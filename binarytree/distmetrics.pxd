#!python
import numpy as np
cimport numpy as np

from typedefs cimport DTYPE_t, ITYPE_t

cdef class DistanceMetric:
    cdef DTYPE_t dist(self, DTYPE_t[:, ::1] X1, ITYPE_t i1,
                      DTYPE_t[:, ::1] X2, ITYPE_t i2)
    cdef DTYPE_t rdist(self, DTYPE_t[:, ::1] X1, ITYPE_t i1,
                      DTYPE_t[:, ::1] X2, ITYPE_t i2)
    cdef DTYPE_t[:, ::1] pdist(self, DTYPE_t[:, ::1])
    cdef DTYPE_t[:, ::1] cdist(self, DTYPE_t[:, ::1], DTYPE_t[:, ::1])

cdef class EuclideanDistance(DistanceMetric):
    pass

cdef class ManhattanDistance(DistanceMetric):
    pass

cdef class MinkowskiDistance(DistanceMetric):
    cdef DTYPE_t p

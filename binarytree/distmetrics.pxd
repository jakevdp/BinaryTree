#!python
import numpy as np
cimport numpy as np

# warning: there will be problems if ITYPE
#  is switched to an unsigned type!
ctypedef np.float64_t DTYPE_t
ctypedef np.intp_t ITYPE_t

cdef class _DistanceMetric:
    cdef DTYPE_t dist(self, DTYPE_t[:, ::1] X1, ITYPE_t i1,
                      DTYPE_t[:, ::1] X2, ITYPE_t i2)
    cdef DTYPE_t[:, ::1] pdist(self, DTYPE_t[:, ::1] )
    cdef DTYPE_t[:, ::1] cdist(self, DTYPE_t[:, ::1], DTYPE_t[:, ::1])

cdef class EuclideanDistance(_DistanceMetric):
    pass

cdef class ManhattanDistance(_DistanceMetric):
    pass

cdef class MinkowskiDistance(_DistanceMetric):
    cdef DTYPE_t p

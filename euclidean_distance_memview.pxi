#======================================================================
# Euclidean Distance Implementation
#======================================================================
from libc.math cimport sqrt

cdef inline DTYPE_t dist(DTYPE_t[:,::1] x1, ITYPE_t i1, 
                         DTYPE_t[:, ::1] x2, ITYPE_t i2):
    cdef DTYPE_t tmp, d = 0
    for j in range(x1.shape[1]):
        tmp = (x1[i1, j] - x2[i2, j])
        d += tmp * tmp
    return sqrt(d)


cdef inline DTYPE_trdist(DTYPE_t[:,::1] x1, ITYPE_t i1, 
                         DTYPE_t[:, ::1] x2, ITYPE_t i2):
    cdef DTYPE_t tmp, d = 0
    for j in range(x1.shape[1]):
        tmp = (x1[i1, j] - x2[i2, j])
        d += tmp * tmp
    return d


cdef inline DTYPE_t dist_to_rdist(DTYPE_t dist):
    return dist * dist

cdef inline DTYPE_t rdist_to_dist(DTYPE_t rdist):
    return sqrt(rdist)

#======================================================================

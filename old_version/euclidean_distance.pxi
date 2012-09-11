#======================================================================
# Euclidean Distance Implementation
#======================================================================
from libc.math cimport sqrt

cdef inline DTYPE_t dist(DTYPE_t* x1, DTYPE_t* x2, ITYPE_t size):
    cdef ITYPE_t j
    cdef DTYPE_t tmp, d = 0
    for j in range(size):
        tmp = (x1[j] - x2[j])
        d += tmp * tmp
    return sqrt(d)


cdef inline DTYPE_t rdist(DTYPE_t* x1, DTYPE_t* x2, ITYPE_t size):
    cdef ITYPE_t j
    cdef DTYPE_t tmp, d = 0
    for j in range(size):
        tmp = (x1[j] - x2[j])
        d += tmp * tmp
    return d


cdef inline DTYPE_t dist_to_rdist(DTYPE_t dist):
    return dist * dist


cdef inline DTYPE_t rdist_to_dist(DTYPE_t rdist):
    return sqrt(rdist)

#======================================================================

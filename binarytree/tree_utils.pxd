from typedefs cimport DTYPE_t, ITYPE_t, DITYPE_t

cdef class MaxHeap:
    cdef DTYPE_t[::1] val
    cdef ITYPE_t[::1] idx

    cpdef wrap(self, DTYPE_t[::1] val, ITYPE_t[::1] idx)
    cpdef DTYPE_t largest(self)
    cpdef ITYPE_t idx_largest(self)
    cpdef push(self, DTYPE_t val, ITYPE_t i_val)

cpdef sort_dist_idx(DTYPE_t[::1] dist, ITYPE_t[::1] idx)
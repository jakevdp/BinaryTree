from typedefs cimport DTYPE_t, ITYPE_t, DITYPE_t

cdef class MaxHeap:
    cdef DTYPE_t[::1] val
    cdef ITYPE_t[::1] idx

    cdef inline DTYPE_t largest(self):
        return self.val[0]

    cdef inline ITYPE_t idx_largest(self):
        return self.idx[0]

    cdef inline void sort(self):
        sort_dist_idx(self.val, self.idx)

    cpdef wrap(self, DTYPE_t[::1] val, ITYPE_t[::1] idx)
    cpdef push(self, DTYPE_t val, ITYPE_t i_val)

cpdef sort_dist_idx(DTYPE_t[::1] dist, ITYPE_t[::1] idx)

cpdef ITYPE_t find_split_dim(DTYPE_t[:, ::1] data,
                             ITYPE_t[::1] indices,
                             ITYPE_t idx_start, ITYPE_t idx_end)

cpdef ITYPE_t partition_indices(DTYPE_t[:, ::1] data,
                                ITYPE_t[::1] indices,
                                ITYPE_t split_dim,
                                ITYPE_t idx_start,
                                ITYPE_t split_index,
                                ITYPE_t idx_end)

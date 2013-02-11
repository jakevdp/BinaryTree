#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
cimport cython
cimport numpy as np

import numpy as np

############################################################
# Define types

# Floating point/data type
ctypedef np.float64_t DTYPE_t

# Index/integer type.
#  WARNING: ITYPE_t must be a signed integer type!!
ctypedef np.intp_t ITYPE_t

# Fused type for certain operations
ctypedef fused DITYPE_t:
    ITYPE_t
    DTYPE_t

# use a hack to determine the associated numpy data types
cdef ITYPE_t idummy
cdef ITYPE_t[:] idummy_view = <ITYPE_t[:1]> &idummy
ITYPE = np.asarray(idummy_view).dtype

cdef DTYPE_t ddummy
cdef DTYPE_t[:] ddummy_view = <DTYPE_t[:1]> &ddummy
DTYPE = np.asarray(ddummy_view).dtype


############################################################
# Utility routines
cdef inline void swap(DITYPE_t[:, ::1] arr, ITYPE_t row,
                      ITYPE_t i1, ITYPE_t i2):
    cdef DITYPE_t tmp = arr[row, i1]
    arr[row, i1] = arr[row, i2]
    arr[row, i2] = tmp


#############################################################
# NeighborsHeap
cdef class NeighborsHeap:
    cdef DTYPE_t[:, ::1] distances
    cdef ITYPE_t[:, ::1] indices

    def __cinit__(self):
        self.distances = np.zeros((1, 1), dtype=DTYPE)
        self.indices = np.zeros((1, 1), dtype=ITYPE)

    def __init__(self, n_pts, n_nbrs):
        self.distances = np.zeros((n_pts, n_nbrs), dtype=DTYPE) + np.inf
        self.indices = np.zeros((n_pts, n_nbrs), dtype=ITYPE)

    cpdef push(self, ITYPE_t row, DTYPE_t val, ITYPE_t i_val):
        cdef DTYPE_t[:, ::1] distances = self.distances
        cdef ITYPE_t[:, ::1] indices = self.indices
        cdef ITYPE_t i, child1, child2, i_swap

        if (row < 0) or (row >= distances.shape[0]):
            raise ValueError("row out of range")

        # if val is larger than the current largest, we end here
        if val >= distances[row, 0]:
            return

        # insert val at position zero
        distances[row, 0] = val
        indices[row, 0] = i_val

        #descend the heap, swapping values until the max heap criterion is met
        i = 0
        while True:
            child1 = 2 * i + 1
            child2 = child1 + 1

            if child1 >= distances.shape[1]:
                break
            elif child2 >= distances.shape[1]:
                if distances[row, child1] > val:
                    i_swap = child1
                else:
                    break
            elif distances[row, child1] >= distances[row, child2]:
                if val < distances[row, child1]:
                    i_swap = child1
                else:
                    break
            else:
                if val < distances[row, child2]:
                    i_swap = child2
                else:
                    break

            distances[row, i] = distances[row, i_swap]
            indices[row, i] = indices[row, i_swap]

            i = i_swap

        distances[row, i] = val
        indices[row, i] = i_val

    cdef _sort(self):
        cdef DTYPE_t[:, ::1] distances = self.distances
        cdef ITYPE_t[:, ::1] indices = self.indices
        cdef ITYPE_t row
        for row in range(distances.shape[0]):
            _simultaneous_sort(distances, indices, row,
                               0, distances.shape[1])

    def get_arrays(self, sort=True):
        if sort:
            self._sort()
        return (self.distances, self.indices)


######################################################################
# simultaneous_sort :
#  this is a recursive quicksort implementation which sorts `distances`
#  and simultaneously performs the same swaps on `indices`.
cdef void _simultaneous_sort(DTYPE_t[:, ::1] distances,
                             ITYPE_t[:, ::1] indices,
                             ITYPE_t row, ITYPE_t lower, ITYPE_t upper):
    # recursive in-place quicksort of the vector distances[row, lower:upper],
    # simultaneously performing the same swaps on the indices array.
    if lower + 1 >= upper:
        return

    cdef DTYPE_t pivot_val
    cdef ITYPE_t pivot_idx, store_idx, i

    # determine new pivot
    pivot_idx = (lower + upper) / 2
    pivot_val = distances[row, pivot_idx]
    store_idx = lower
    swap(distances, row, pivot_idx, upper - 1)
    swap(indices, row, pivot_idx, upper - 1)
    for i in range(lower, upper - 1):
        if distances[row, i] < pivot_val:
            swap(distances, row, i, store_idx)
            swap(indices, row, i, store_idx)
            store_idx += 1
    swap(distances, row, store_idx, upper - 1)
    swap(indices, row, store_idx, upper - 1)
    pivot_idx = store_idx

    # recursively sort each side of the pivot
    if lower + 1 < pivot_idx:
        _simultaneous_sort(distances, indices, row, lower, pivot_idx)
    if pivot_idx + 2 < upper:
        _simultaneous_sort(distances, indices, row, pivot_idx + 1, upper)


############################################################
# Python access functions

def load_heap(DTYPE_t[:, ::1] X, ITYPE_t k):
    """test fully loading the heap"""
    assert k <= X.shape[1]
    cdef NeighborsHeap heap = NeighborsHeap(X.shape[0], k)
    cdef ITYPE_t i, j
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            heap.push(i, X[i, j], j)
    return heap.get_arrays()
            

def simultaneous_sort(DTYPE_t[:, ::1] distances, ITYPE_t[:, ::1] indices):
    """In-place simultaneous sort the given row of the arrays"""
    assert distances.shape[0] == indices.shape[0]
    assert distances.shape[1] == indices.shape[1]

    cdef ITYPE_t row
    for row in range(distances.shape[0]):
        _simultaneous_sort(distances, indices, row, 0, distances.shape[1])

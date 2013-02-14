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
cdef inline void swap(DITYPE_t* arr, ITYPE_t i1, ITYPE_t i2):
    cdef DITYPE_t tmp = arr[i1]
    arr[i1] = arr[i2]
    arr[i2] = tmp


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
        cdef ITYPE_t i, ic1, ic2, i_swap
        cdef ITYPE_t size = self.distances.shape[1]
        cdef DTYPE_t* dist_arr = &self.distances[row, 0]
        cdef ITYPE_t* ind_arr = &self.indices[row, 0]

        # check if val should be in heap
        if val > dist_arr[0]:
            return

        # insert val at position zero
        dist_arr[0] = val
        ind_arr[0] = i_val

        #descend the heap, swapping values until the max heap criterion is met
        i = 0
        while True:
            ic1 = 2 * i + 1
            ic2 = ic1 + 1

            if ic1 >= size:
                break
            elif ic2 >= size:
                if dist_arr[ic1] > val:
                    i_swap = ic1
                else:
                    break
            elif dist_arr[ic1] >= dist_arr[ic2]:
                if val < dist_arr[ic1]:
                    i_swap = ic1
                else:
                    break
            else:
                if val < dist_arr[ic2]:
                    i_swap = ic2
                else:
                    break

            dist_arr[i] = dist_arr[i_swap]
            ind_arr[i] = ind_arr[i_swap]

            i = i_swap

        dist_arr[i] = val
        ind_arr[i] = i_val

    cdef _sort(self):
        cdef DTYPE_t[:, ::1] distances = self.distances
        cdef ITYPE_t[:, ::1] indices = self.indices
        cdef ITYPE_t row
        for row in range(distances.shape[0]):
            _simultaneous_sort(&distances[row, 0],
                               &indices[row, 0],
                               distances.shape[1])

    def get_arrays(self, sort=True):
        if sort:
            self._sort()
        return (self.distances, self.indices)


######################################################################
# _simultaneous_sort :
#  this is a recursive quicksort implementation which sorts `dist` and
#  simultaneously performs the same swaps on `idx`.
cdef void _simultaneous_sort(DTYPE_t* dist, ITYPE_t* idx, ITYPE_t size):
    cdef ITYPE_t pivot_idx, store_idx, i
    cdef DTYPE_t pivot_val

    if size <= 1:
        return

    # determine new pivot
    pivot_idx = size / 2
    pivot_val = dist[pivot_idx]
    store_idx = 0

    swap(dist, pivot_idx, size - 1)
    swap(idx, pivot_idx, size - 1)

    for i in range(size - 1):
        if dist[i] < pivot_val:
            swap(dist, i, store_idx)
            swap(idx, i, store_idx)
            store_idx += 1
    swap(dist, store_idx, size - 1)
    swap(idx, store_idx, size - 1)
    pivot_idx = store_idx

    # recursively sort each side of the pivot
    if pivot_idx > 1:
        _simultaneous_sort(dist, idx, pivot_idx)
    if pivot_idx + 2 < size:
        _simultaneous_sort(dist + pivot_idx + 1,
                           idx + pivot_idx + 1,
                           size - pivot_idx - 1)


############################################################
# Python access functions for benchmarking

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
    """In-place simultaneous sort the given row of the arrays
    
    This python wrapper exists primarily to enable unit testing
    of the _simultaneous_sort C routine.
    """
    assert distances.shape[0] == indices.shape[0]
    assert distances.shape[1] == indices.shape[1]
    cdef ITYPE_t row
    for row in range(distances.shape[0]):
        _simultaneous_sort(&distances[row, 0],
                           &indices[row, 0],
                           distances.shape[1])

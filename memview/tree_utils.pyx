#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
cimport cython
cimport numpy as np

from libc.math cimport fmax, fmin, fabs

import numpy as np
from typedefs import DTYPE, ITYPE

######################################################################
# Max-heap for keeping track of neighbors
#
#  This is a basic implementation of a fixed-size binary max-heap.
#
#  The root node is at heap[0].  The two child nodes of node i are at
#  (2 * i + 1) and (2 * i + 2).
#  The parent node of node i is node floor((i-1)/2); node 0 has no parent.
#  A max heap has heap[i] >= max(heap[2 * i + 1], heap[2 * i + 2])
#  for all valid indices.
#
#  In this implementation, an empty heap should be full of infinities
#
@cython.final
cdef class MaxHeap:
    def __init__(self, size=1):
        self.val = np.zeros(size, dtype=DTYPE) + np.inf
        self.idx = np.zeros(size, dtype=ITYPE)

    def get_arrays(self, sort=False):
        if sort:
            sort_dist_idx(self.val, self.idx)
        return np.asarray(self.val), np.asarray(self.idx)

    cpdef wrap(self, DTYPE_t[::1] val, ITYPE_t[::1] idx):
        self.val = val
        self.idx = idx

        if self.val.shape[0] != self.idx.shape[0]:
            raise ValueError("val and idx shapes should match")

    cpdef push(self, DTYPE_t val, ITYPE_t i_val):
        cdef ITYPE_t i, ic1, ic2, i_swap

        # check if val is larger than the current largest
        if val > self.val[0]:
            return

        # insert val at position zero
        self.val[0] = val
        self.idx[0] = i_val

        #descend the heap, swapping values until the max heap criterion is met
        i = 0
        while 1:
            ic1 = 2 * i + 1
            ic2 = ic1 + 1

            if ic1 >= self.val.shape[0]:
                break
            elif ic2 >= self.val.shape[0]:
                if self.val[ic1] > val:
                    i_swap = ic1
                else:
                    break
            elif self.val[ic1] >= self.val[ic2]:
                if val < self.val[ic1]:
                    i_swap = ic1
                else:
                    break
            else:
                if val < self.val[ic2]:
                    i_swap = ic2
                else:
                    break

            self.val[i] = self.val[i_swap]
            self.idx[i] = self.idx[i_swap]

            i = i_swap

        self.val[i] = val
        self.idx[i] = i_val


######################################################################
# sort_dist_idx :
#  this is a recursive quicksort implementation which sorts `dist` and
#  simultaneously performs the same swaps on `idx`.  The result is
#  identical to
#    i = np.argsort(dist)
#    dist = dist[i]
#    idx = idx[i]
cdef inline void swap(DITYPE_t[::1] arr, ITYPE_t i1, ITYPE_t i2):
    cdef DITYPE_t tmp = arr[i1]
    arr[i1] = arr[i2]
    arr[i2] = tmp


cpdef sort_dist_idx(DTYPE_t[::1] dist, ITYPE_t[::1] idx):
    if dist.shape[0] != idx.shape[0]:
        raise ValueError('dist and ind should have matching shapes')
    if dist.shape[0] > 1:
        _sort_dist_idx(dist, idx, 0, dist.shape[0])


cdef void _sort_dist_idx(DTYPE_t[::1] dist, ITYPE_t[::1] idx,
                         ITYPE_t lower, ITYPE_t upper):
    cdef DTYPE_t pivot_val
    cdef ITYPE_t pivot_idx, store_idx, i

    # determine new pivot
    pivot_idx = (lower + upper) / 2
    pivot_val = dist[pivot_idx]
    store_idx = lower
    swap(dist, pivot_idx, upper - 1)
    swap(idx, pivot_idx, upper - 1)
    for i in range(lower, upper - 1):
        if dist[i] < pivot_val:
            swap(dist, i, store_idx)
            swap(idx, i, store_idx)
            store_idx += 1
    swap(dist, store_idx, upper - 1)
    swap(idx, store_idx, upper - 1)
    pivot_idx = store_idx

    # recursively sort each side of the pivot
    if lower + 1 < pivot_idx:
        _sort_dist_idx(dist, idx, lower, pivot_idx)
    if pivot_idx + 2 < upper:
        _sort_dist_idx(dist, idx, pivot_idx + 1, upper)


######################################################################
# find_split_dim:
#  this computes the equivalent of the following:
#  j_max = np.argmax(np.max(data[indices[idx_start:idx_end]], 0) -
#                    np.min(data[indices[idx_start:idx_end]], 0))
cpdef ITYPE_t find_split_dim(DTYPE_t[:, ::1] data,
                             ITYPE_t[::1] indices,
                             ITYPE_t idx_start, ITYPE_t idx_end):
    cdef DTYPE_t min_val, max_val, val, spread, max_spread
    cdef ITYPE_t i, j, j_max

    if data.shape[0] != indices.shape[0]:
        raise ValueError('data and indices sizes do not match')

    j_max = 0
    max_spread = 0

    for j in range(data.shape[1]):
        min_val = max_val = data[indices[idx_start], j]
        for i in range(idx_start, idx_end):
            val = data[indices[i], j]
            max_val = fmax(max_val, val)
            min_val = fmin(min_val, val)
        spread = max_val - min_val
        if spread > max_spread:
            max_spread = spread
            j_max = j
    return j_max


######################################################################
# partition_indices:
#  in-place modification of the sub-array indices[idx_start:idx_end]
#  Such that upon return (assuming numpy-style fancy indexing)
#    np.all(data[indices[idx_start:split_index], split_dim]
#           <= data[indices[split_index], split_dim])
#  and
#    np.all(data[indices[split_index], split_dim]
#           <= data[indices[split_index:idx_end], split_dim])
#  will hold.  The algorithm amounts to a partial quicksort.
#  An integer is returned to be compatible with cpdef
cpdef ITYPE_t partition_indices(DTYPE_t[:, ::1] data,
                                ITYPE_t[::1] indices,
                                ITYPE_t split_dim,
                                ITYPE_t idx_start,
                                ITYPE_t split_index,
                                ITYPE_t idx_end):
    cdef ITYPE_t left, right, midindex, i
    cdef DTYPE_t d1, d2
    left = idx_start
    right = idx_end - 1

    if data.shape[0] != indices.shape[0]:
        raise ValueError('data and indices sizes do not match')

    if ((split_index < idx_start) or (split_index >= idx_end)):
        raise ValueError("split index out of range")

    while True:
        midindex = left
        for i in range(left, right):
            d1 = data[indices[i], split_dim]
            d2 = data[indices[right], split_dim]
            if d1 < d2:
                swap(indices, i, midindex)
                midindex += 1
        swap(indices, midindex, right)
        if midindex == split_index:
            break
        elif midindex < split_index:
            left = midindex + 1
        else:
            right = midindex - 1

    return 0

#!python
cimport cython
cimport numpy as np
from distmetrics cimport DTYPE_t, ITYPE_t
from distmetrics import DTYPE, ITYPE

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

cdef class MaxHeap:
    cdef DTYPE_t[::1] val
    cdef ITYPE_t[::1] idx

    def __init__(self, val, idx):
        self.val = val
        self.idx = idx

        if self.val.shape[0] != self.idx.shape[0]:
            raise ValueError("val and idx shapes should match")

    cpdef DTYPE_t largest(self):
        return self.val[0]

    cpdef ITYPE_t idx_largest(self):
        return self.idx[0]

    cpdef insert(self, DTYPE_t val, ITYPE_t i_val):
        cdef ITYPE_t i, ic1, ic2, i_tmp
        cdef DTYPE_t d_tmp

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
#  simultaneously performs the same swaps on `idx`.
ctypedef fused DITYPE_t:
    ITYPE_t
    DTYPE_t

cdef inline void swap(DITYPE_t[::1] arr, ITYPE_t i1, ITYPE_t i2):
    cdef DITYPE_t tmp = arr[i1]
    arr[i1] = arr[i2]
    arr[i2] = tmp

cpdef sort_dist_idx(DTYPE_t[::1] dist, ITYPE_t[::1] idx):
    cdef ITYPE_t pivot_idx, store_idx, i
    cdef DTYPE_t pivot_val
    cdef ITYPE_t k = dist.shape[0]

    if k > 1:
        # determine new pivot
        pivot_idx = k / 2
        pivot_val = dist[pivot_idx]
        store_idx = 0
                         
        swap(dist, pivot_idx, k - 1)
        swap(idx, pivot_idx, k - 1)

        for i in range(k - 1):
            if dist[i] < pivot_val:
                swap(dist, i, store_idx)
                swap(idx, i, store_idx)
                store_idx += 1
        swap(dist, store_idx, k - 1)
        swap(idx, store_idx, k - 1)
        pivot_idx = store_idx

        # recursively sort each side of the pivot
        sort_dist_idx(dist[:pivot_idx], idx[:pivot_idx])
        sort_dist_idx(dist[pivot_idx + 1:], idx[pivot_idx + 1:])

"""Binary Tree

This is the Abstract Base Class for the Ball Tree and KD Tree
"""

#!python
import numpy as np
from sklearn.utils import array2d

cimport numpy as np
cimport cython

from distmetrics cimport DistanceMetric

#######################################################################
# Type definitions
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

# warning: there will be problems if ITYPE
#  is switched to an unsigned type!
ITYPE = np.intp
ctypedef np.intp_t ITYPE_t

# explicitly define infinity
cdef DTYPE_t INF = np.inf

######################################################################
# newObj function
#  this is a helper function for pickling
def newObj(obj):
    return obj.__new__(obj)

######################################################################
# NodeData struct
#  used to keep track of information for individual nodes
#  defined in binary_tree.pxd
# get the numpy dtype corresponding to this
cdef struct NodeData_t:
    ITYPE_t idx_start
    ITYPE_t idx_end
    int is_leaf
    DTYPE_t radius

# use a dummy variable to determine the python data type
cdef NodeData_t dummy
cdef NodeData_t[:] dummy_view = <NodeData_t[:1]> &dummy
NodeData = np.asarray(dummy_view).dtype

######################################################################
# BinaryTree Abstract Base Class
cdef class _BinaryTree:
    """Abstract base class for binary tree objects"""
    cdef readonly DTYPE_t[:, ::1] data
    cdef ITYPE_t[:, ::1] idx_array
    cdef NodeData_t[::1] node_data_arr

    cdef ITYPE_t leaf_size
    cdef ITYPE_t n_levels
    cdef ITYPE_t n_nodes

    cdef HeapData heap
    cdef DistanceMetric dm



######################################################################
# Max-heap for keeping track of neighbors
#
#  This is a basic implementation of a fixed-size binary max-heap.
#  It can be used in place of priority_queue to keep track of the
#  k-nearest neighbors in a query.  The implementation is faster than
#  priority_queue for a very large number of neighbors (k > 50 or so).
#  The implementation is slower than priority_queue for fewer neighbors.
#  The other disadvantage is that for max_heap, the indices/distances must
#  be sorted upon completion of the query.  In priority_queue, the indices
#  and distances are sorted without an extra call.
#
#  The root node is at heap[0].  The two child nodes of node i are at
#  (2 * i + 1) and (2 * i + 2).
#  The parent node of node i is node floor((i-1)/2).  Node 0 has no parent.
#  A max heap has (heap[i] >= heap[2 * i + 1]) and (heap[i] >= heap[2 * i + 2])
#  for all valid indices.
#
#  In this implementation, an empty heap should be full of infinities
#

cdef struct HeapData:
    DTYPE_t* val
    ITYPE_t* idx
    ITYPE_t size


cdef inline void heap_init(HeapData* heapdata, DTYPE_t* val,
                           ITYPE_t* idx, ITYPE_t size):
    heapdata.val = val
    heapdata.idx = idx
    heapdata.size = size


cdef inline int heap_needs_final_sort(HeapData* heapdata):
    return 1


cdef inline DTYPE_t heap_largest(HeapData* heapdata):
    return heapdata.val[0]


cdef inline ITYPE_t heap_idx_largest(HeapData* heapdata):
    return heapdata.idx[0]


cdef inline void heap_insert(HeapData* heapdata, DTYPE_t val, ITYPE_t i_val):
    cdef ITYPE_t i, ic1, ic2, i_tmp
    cdef DTYPE_t d_tmp

    # check if val should be in heap
    if val > heapdata.val[0]:
        return

    # insert val at position zero
    heapdata.val[0] = val
    heapdata.idx[0] = i_val

    #descend the heap, swapping values until the max heap criterion is met
    i = 0
    while 1:
        ic1 = 2 * i + 1
        ic2 = ic1 + 1

        if ic1 >= heapdata.size:
            break
        elif ic2 >= heapdata.size:
            if heapdata.val[ic1] > val:
                i_swap = ic1
            else:
                break
        elif heapdata.val[ic1] >= heapdata.val[ic2]:
            if val < heapdata.val[ic1]:
                i_swap = ic1
            else:
                break
        else:
            if val < heapdata.val[ic2]:
                i_swap = ic2
            else:
                break

        heapdata.val[i] = heapdata.val[i_swap]
        heapdata.idx[i] = heapdata.idx[i_swap]

        i = i_swap

    heapdata.val[i] = val
    heapdata.idx[i] = i_val


######################################################################
# Helper functions for building and querying
#
cdef ITYPE_t find_split_dim(DTYPE_t* data,
                            ITYPE_t* node_indices,
                            ITYPE_t n_features,
                            ITYPE_t n_points):
    # this computes the following
    # j_max = np.argmax(np.max(data, 0) - np.min(data, 0))
    cdef DTYPE_t min_val, max_val, val, spread, max_spread
    cdef ITYPE_t i, j, j_max

    j_max = 0
    max_spread = 0

    for j in range(n_features):
        max_val = data[node_indices[0] * n_features + j]
        min_val = max_val
        for i in range(1, n_points):
            val = data[node_indices[i] * n_features + j]
            max_val = fmax(max_val, val)
            min_val = fmin(min_val, val)
        spread = max_val - min_val
        if spread > max_spread:
            max_spread = spread
            j_max = j
    return j_max


@cython.profile(False)
cdef inline void iswap(ITYPE_t* arr, ITYPE_t i1, ITYPE_t i2):
    cdef ITYPE_t tmp = arr[i1]
    arr[i1] = arr[i2]
    arr[i2] = tmp


@cython.profile(False)
cdef inline void dswap(DTYPE_t* arr, ITYPE_t i1, ITYPE_t i2):
    cdef DTYPE_t tmp = arr[i1]
    arr[i1] = arr[i2]
    arr[i2] = tmp


cdef void partition_indices(DTYPE_t* data,
                            ITYPE_t* node_indices,
                            ITYPE_t split_dim,
                            ITYPE_t split_index,
                            ITYPE_t n_features,
                            ITYPE_t n_points):
    # partition_indices will modify the array node_indices between
    # indices 0 and n_points.  Upon return (assuming numpy-style slicing)
    #   data[node_indices[0:split_index], split_dim]
    #     <= data[node_indices[split_index], split_dim]
    # and
    #   data[node_indices[split_index], split_dim]
    #     <= data[node_indices[split_index:n_points], split_dim]
    # will hold.  The algorithm amounts to a partial quicksort
    cdef ITYPE_t left, right, midindex, i
    cdef DTYPE_t d1, d2
    left = 0
    right = n_points - 1

    while True:
        midindex = left
        for i in range(left, right):
            d1 = data[node_indices[i] * n_features + split_dim]
            d2 = data[node_indices[right] * n_features + split_dim]
            if d1 < d2:
                iswap(node_indices, i, midindex)
                midindex += 1
        iswap(node_indices, midindex, right)
        if midindex == split_index:
            break
        elif midindex < split_index:
            left = midindex + 1
        else:
            right = midindex - 1


######################################################################
# sort_dist_idx :
#  this is a recursive quicksort implementation which sorts `dist` and
#  simultaneously performs the same swaps on `idx`.
cdef void sort_dist_idx(DTYPE_t* dist, ITYPE_t* idx, ITYPE_t k):
    cdef ITYPE_t pivot_idx, store_idx, i
    cdef DTYPE_t pivot_val

    if k > 1:
        #-- determine pivot -----------
        pivot_idx = k / 2
        pivot_val = dist[pivot_idx]
        store_idx = 0
                         
        dswap(dist, pivot_idx, k - 1)
        iswap(idx, pivot_idx, k - 1)

        for i in range(k - 1):
            if dist[i] < pivot_val:
                dswap(dist, i, store_idx)
                iswap(idx, i, store_idx)
                store_idx += 1
        dswap(dist, store_idx, k - 1)
        iswap(idx, store_idx, k - 1)
        pivot_idx = store_idx
        #------------------------------

        sort_dist_idx(dist, idx, pivot_idx)

        sort_dist_idx(dist + pivot_idx + 1,
                      idx + pivot_idx + 1,
                      k - pivot_idx - 1)

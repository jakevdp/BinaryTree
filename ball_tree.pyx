#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

cimport cython
cimport numpy as np
from libc.math cimport fmax, fmin, fabs, sqrt, exp, cos

import numpy as np
import warnings

######################################################################
# Define types

# Floating point/data type
ctypedef np.float64_t DTYPE_t

# Index/integer type.
#  WARNING: ITYPE_t must be a signed integer type or you will have a bad time!
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

cdef DTYPE_t INF = np.inf
cdef DTYPE_t PI = np.pi
cdef DTYPE_t ROOT_2PI = sqrt(2 * PI)

######################################################################
# Inline distance functions
#
#  We use these for the default (euclidean) case so that they can
#  be inlined.  This leads to much faster computation for this case.
cdef inline DTYPE_t euclidean_dist(DTYPE_t* x1, DTYPE_t* x2,
                                   ITYPE_t size):
    cdef DTYPE_t tmp, d=0
    for j in range(size):
        tmp = x1[j] - x2[j]
        d += tmp * tmp
    return sqrt(d)

cdef inline DTYPE_t euclidean_rdist(DTYPE_t* x1, DTYPE_t* x2,
                                    ITYPE_t size):
    cdef DTYPE_t tmp, d=0
    for j in range(size):
        tmp = x1[j] - x2[j]
        d += tmp * tmp
    return d

cdef inline DTYPE_t euclidean_dist_to_rdist(DTYPE_t dist):
    return dist * dist

cdef inline DTYPE_t euclidean_rdist_to_dist(DTYPE_t dist):
    return sqrt(dist)

cdef DTYPE_t[:, ::1] euclidean_cdist(DTYPE_t[:, ::1] X, DTYPE_t[:, ::1] Y):
    if X.shape[1] != Y.shape[1]:
        raise ValueError('X and Y must have the same second dimension')

    cdef DTYPE_t[:, ::1] D = np.zeros((X.shape[0], Y.shape[0]), dtype=DTYPE)

    for i1 in range(X.shape[0]):
        for i2 in range(Y.shape[0]):
            D[i1, i2] = euclidean_dist(&X[i1, 0], &Y[i2, 0], X.shape[1])
    return D


######################################################################
# Kernel functions
#
# Note: Kernels assume dist is non-negative and and h is positive
#       All kernel functions are normalized such that K(0, h) = 1.
#       The fully normalized kernel is:
#         kernel_norm(h, kernel) * compute_kernel(dist, h, kernel)
cdef enum KernelType:
    GAUSSIAN_KERNEL = 1
    TOPHAT_KERNEL = 2
    EPANECHNIKOV_KERNEL = 3
    EXPONENTIAL_KERNEL = 4
    LINEAR_KERNEL = 5
    COSINE_KERNEL = 6

cdef inline DTYPE_t gaussian_kernel(DTYPE_t dist, DTYPE_t h):
    return exp(-0.5 * (dist * dist) / (h * h))

cdef inline DTYPE_t tophat_kernel(DTYPE_t dist, DTYPE_t h):
    if dist < h:
        return 1.0
    else:
        return 0.0

cdef inline DTYPE_t epanechnikov_kernel(DTYPE_t dist, DTYPE_t h):
    if dist < h:
        return 1.0 - (dist * dist) / (h * h)
    else:
        return 0.0

cdef inline DTYPE_t exponential_kernel(DTYPE_t dist, DTYPE_t h):
    if dist < h:
        return exp(-dist / h)
    else:
        return 0.0

cdef inline DTYPE_t linear_kernel(DTYPE_t dist, DTYPE_t h):
    if dist < h:
        return 1 - dist / h
    else:
        return 0.0

cdef inline DTYPE_t cosine_kernel(DTYPE_t dist, DTYPE_t h):
    if dist < h:
        return cos(0.5 * PI * dist / h)
    else:
        return 0.0


cdef inline DTYPE_t compute_kernel(DTYPE_t dist, DTYPE_t h,
                                   KernelType kernel):
    if kernel == GAUSSIAN_KERNEL:
        return gaussian_kernel(dist, h)
    elif kernel == TOPHAT_KERNEL:
        return tophat_kernel(dist, h)
    if kernel == EPANECHNIKOV_KERNEL:
        return epanechnikov_kernel(dist, h)
    elif kernel == EXPONENTIAL_KERNEL:
        return exponential_kernel(dist, h)
    if kernel == LINEAR_KERNEL:
        return linear_kernel(dist, h)
    elif kernel == COSINE_KERNEL:
        return cosine_kernel(dist, h)


cdef inline DTYPE_t kernel_norm(DTYPE_t h, KernelType kernel):
    if kernel == GAUSSIAN_KERNEL:
        return 1. / (h * ROOT_2PI)
    elif kernel == TOPHAT_KERNEL:
        return 0.5 / h
    if kernel == EPANECHNIKOV_KERNEL:
        return 0.75 / h
    elif kernel == EXPONENTIAL_KERNEL:
        return 0.5 / h
    if kernel == LINEAR_KERNEL:
        return 1.0 / h
    elif kernel == COSINE_KERNEL:
        return 0.25 * PI / h


######################################################################
# Distance Metric Classes

#------------------------------------------------------------
# DistanceMetric base class
cdef class DistanceMetric:
    cdef DTYPE_t p

    @classmethod
    def get_metric(cls, metric, **kwargs):
        if isinstance(metric, DistanceMetric):
            return metric
        elif isinstance(metric, type) and issubclass(metric, DistanceMetric):
            return metric(**kwargs)
        elif metric in [None, 'euclidean', 'l2']:
            return EuclideanDistance()
        elif metric in ['manhattan', 'cityblock', 'l1']:
            return ManhattanDistance()
        elif metric in ['minkowski', 'p']:
            p = kwargs.get('p', 2)
            if p == 1:
                return ManhattanDistance()
            if p == 2:
                return EuclideanDistance()
            else:
                return MinkowskiDistance(p)
        else:
            raise ValueError('metric = "%s" not recognized' % str(metric))

    def __init__(self, **kwargs):
        if self.__class__ is DistanceMetric:
            raise NotImplementedError("DistanceMetric is an abstract class")

    cdef DTYPE_t dist(self, DTYPE_t* x1, DTYPE_t* x2, ITYPE_t size):
        return -999

    cdef DTYPE_t rdist(self, DTYPE_t* x1, DTYPE_t* x2, ITYPE_t size):
        return self.dist(x1, x2, size)

    cdef DTYPE_t[:, ::1] pdist(self, DTYPE_t[:, ::1] X):
        cdef ITYPE_t i1, i2
        cdef DTYPE_t[:, ::1] D = np.zeros((X.shape[0], X.shape[0]),
                                          dtype=DTYPE, order='C')
        for i1 in range(X.shape[0]):
            for i2 in range(i1, X.shape[0]):
                D[i1, i2] = self.dist(&X[i1, 0], &X[i2, 0], X.shape[1])
                D[i2, i1] = D[i1, i2]
        return D

    cdef DTYPE_t[:, ::1] cdist(self, DTYPE_t[:, ::1] X, DTYPE_t[:, ::1] Y):
        cdef ITYPE_t i1, i2
        if X.shape[1] != Y.shape[1]:
            raise ValueError('X and Y must have the same second dimension')

        cdef DTYPE_t[:, ::1] D = np.zeros((X.shape[0], Y.shape[0]),
                                          dtype=DTYPE, order='C')

        for i1 in range(X.shape[0]):
            for i2 in range(Y.shape[0]):
                D[i1, i2] = self.dist(&X[i1, 0], &Y[i2, 0], X.shape[1])
        return D

    cdef DTYPE_t rdist_to_dist(self, DTYPE_t rdist):
        return rdist

    cdef DTYPE_t dist_to_rdist(self, DTYPE_t dist):
        return dist

    def rdist_to_dist_arr(self, rdist):
        return rdist

    def dist_to_rdist_arr(self, dist):
        return dist

    def pairwise(self, X, Y=None):
        X = np.asarray(X, dtype=DTYPE)
        if Y is None:
            D = self.pdist(X)
        else:
            Y = np.asarray(Y, dtype=DTYPE)
            D = self.cdist(X, Y)
        return np.asarray(D)


#------------------------------------------------------------
# EuclideanDistance specialization
cdef class EuclideanDistance(DistanceMetric):
    def __init__(self):
        self.p = 2

    cdef inline DTYPE_t dist(self, DTYPE_t* x1, DTYPE_t* x2, ITYPE_t size):
        return euclidean_dist(x1, x2, size)

    cdef inline DTYPE_t rdist(self, DTYPE_t* x1, DTYPE_t* x2, ITYPE_t size):
        return euclidean_rdist(x1, x2, size)

    cdef inline DTYPE_t rdist_to_dist(self, DTYPE_t rdist):
        return sqrt(rdist)

    cdef inline DTYPE_t dist_to_rdist(self, DTYPE_t dist):
        return dist * dist

    def rdist_to_dist_arr(self, rdist):
        return np.sqrt(rdist)

    def dist_to_rdist_arr(self, dist):
        return dist ** 2


cdef class ManhattanDistance(DistanceMetric):
    def __init__(self):
        self.p = 1

    cdef inline DTYPE_t dist(self, DTYPE_t* x1, DTYPE_t* x2, ITYPE_t size):
        cdef DTYPE_t d = 0
        for j in range(size):
            d += fabs(x1[j] - x2[j])
        return d


cdef class MinkowskiDistance(DistanceMetric):
    def __init__(self, p):
        self.p = p

    cdef inline DTYPE_t dist(self, DTYPE_t* x1, DTYPE_t* x2, ITYPE_t size):
        cdef DTYPE_t d = 0
        for j in range(size):
            d += pow(fabs(x1[j] - x2[j]), self.p)
        return pow(d, 1. / self.p)

    cdef inline DTYPE_t rdist(self, DTYPE_t* x1, DTYPE_t* x2, ITYPE_t size):
        cdef DTYPE_t d=0
        for j in range(size):
            d += pow(fabs(x1[j] - x2[j]), self.p)
        return d

    cdef inline DTYPE_t rdist_to_dist(self, DTYPE_t rdist):
        return pow(rdist, 1. / self.p)

    cdef inline DTYPE_t dist_to_rdist(self, DTYPE_t dist):
        return pow(dist, self.p)

    def rdist_to_dist_arr(self, rdist):
        return rdist ** (1. / self.p)

    def dist_to_rdist_arr(self, dist):
        return dist ** self.p


######################################################################
# Tree Utility Routines
cdef inline void swap(DITYPE_t* arr, ITYPE_t i1, ITYPE_t i2):
    cdef DITYPE_t tmp = arr[i1]
    arr[i1] = arr[i2]
    arr[i2] = tmp

#------------------------------------------------------------
# NeighborsHeap
#  max-heap structure to keep track of distances and indices of neighbors
cdef class NeighborsHeap:
    cdef DTYPE_t[:, ::1] distances
    cdef ITYPE_t[:, ::1] indices

    def __cinit__(self):
        self.distances = np.zeros((1, 1), dtype=DTYPE)
        self.indices = np.zeros((1, 1), dtype=ITYPE)

    def __init__(self, n_pts, n_nbrs):
        self.distances = np.zeros((n_pts, n_nbrs), dtype=DTYPE) + np.inf
        self.indices = np.zeros((n_pts, n_nbrs), dtype=ITYPE)

    def get_arrays(self, sort=True):
        if sort:
            self._sort()
        return map(np.asarray, (self.distances, self.indices))

    cdef inline DTYPE_t largest(self, ITYPE_t row):
        return self.distances[row, 0]

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


#------------------------------------------------------------
# simultaneous_sort :
#  this is a recursive quicksort implementation which sorts `distances`
#  and simultaneously performs the same swaps on `indices`.
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


#------------------------------------------------------------
# find_split_dim:
#  this computes the equivalent of the 
#  j_max = np.argmax(np.max(data, 0) - np.min(data, 0))
cdef ITYPE_t find_split_dim(DTYPE_t* data,
                            ITYPE_t* node_indices,
                            ITYPE_t n_features,
                            ITYPE_t n_points):
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


#------------------------------------------------------------
# partition_indices will modify the array node_indices between
# indices 0 and n_points.  Upon return (assuming numpy-style slicing)
#   data[node_indices[0:split_index], split_dim]
#     <= data[node_indices[split_index], split_dim]
# and
#   data[node_indices[split_index], split_dim]
#     <= data[node_indices[split_index:n_points], split_dim]
# will hold.  The algorithm amounts to a partial quicksort
cdef void partition_indices(DTYPE_t* data,
                            ITYPE_t* node_indices,
                            ITYPE_t split_dim,
                            ITYPE_t split_index,
                            ITYPE_t n_features,
                            ITYPE_t n_points):
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
                swap(node_indices, i, midindex)
                midindex += 1
        swap(node_indices, midindex, right)
        if midindex == split_index:
            break
        elif midindex < split_index:
            left = midindex + 1
        else:
            right = midindex - 1

######################################################################
# NodeHeap : max heap used to keep track of nodes during
#            breadth-first query

cdef struct NodeHeapData_t:
    DTYPE_t val
    ITYPE_t i1
    ITYPE_t i2

# use a dummy variable to determine the python data type
cdef NodeHeapData_t ndummy
cdef NodeHeapData_t[:] ndummy_view = <NodeHeapData_t[:1]> &ndummy
NodeHeapData = np.asarray(ndummy_view).dtype


cdef inline void swap_nodes(NodeHeapData_t* arr, ITYPE_t i1, ITYPE_t i2):
    cdef NodeHeapData_t tmp = arr[i1]
    arr[i1] = arr[i2]
    arr[i2] = tmp


cdef class NodeHeap:
    """NodeHeap

    This is a min-heap implementation for keeping track of nodes
    during a breadth-first search.

    The heap is an array which is maintained so that the min heap condition
    is met:
       heap[i].val < min(heap[2 * i + 1].val, heap[2 * i + 2].val)
    """
    cdef NodeHeapData_t[::1] data
    cdef ITYPE_t n
    def __cinit__(self):
        self.data = np.zeros(0, dtype=NodeHeapData)

    def __init__(self, size_guess=100):
        self.data = np.zeros(size_guess, dtype=NodeHeapData)
        cdef NodeHeapData_t* data_arr = &self.data[0]
        for i in range(self.data.shape[0]):
            data_arr[i].val = INF
        self.n = 0

    cdef void resize(self, ITYPE_t new_size):
        cdef NodeHeapData_t *data_arr, *new_data_arr
        cdef ITYPE_t i
        cdef ITYPE_t size = self.data.shape[0]
        cdef NodeHeapData_t[::1] new_data = np.zeros(new_size,
                                                     dtype=NodeHeapData)

        if size > 0 and new_size > 0:
            data_arr = &self.data[0]
            new_data_arr = &new_data[0]
            for i in range(min(size, new_size)):
                new_data_arr[i] = data_arr[i]
            
        if new_size > size:
            new_data_arr = &new_data[0]
            for i in range(size, new_size):
                new_data[i].val = -INF

        self.data = new_data

    cdef void push(self, NodeHeapData_t data):
        cdef ITYPE_t i, i_parent
        cdef NodeHeapData_t* data_arr
        self.n += 1
        if self.n > self.data.shape[0]:
            self.resize(2 * self.n)

        # put the new element at the end,
        # and then perform swaps until the heap is in order
        data_arr = &self.data[0]
        i = self.n - 1
        data_arr[i] = data
        
        while i > 0:
            i_parent = (i - 1) // 2
            if data_arr[i_parent].val <= data_arr[i].val:
                break
            else:
                swap_nodes(data_arr, i, i_parent)
                i = i_parent

    cdef NodeHeapData_t peek(self):
        return self.data[0]

    cdef NodeHeapData_t pop(self):
        if self.n == 0:
            raise ValueError('cannot pop on empty heap')

        cdef ITYPE_t i, i_child1, i_child2, i_swap
        cdef NodeHeapData_t* data_arr = &self.data[0]
        cdef NodeHeapData_t popped_element = data_arr[0]

        # pop off the first element, move the last element to the front,
        # and then perform swaps until the heap is back in order
        data_arr[0] = data_arr[self.n - 1]
        data_arr[self.n - 1].val = INF
        self.n -= 1

        i = 0

        while (i < self.n):
            i_child1 = 2 * i + 1
            i_child2 = 2 * i + 2
            i_swap = 0

            if i_child2 < self.n:
                if data_arr[i_child1].val <= data_arr[i_child2].val:
                    i_swap = i_child1
                else:
                    i_swap = i_child2
            elif i_child1 < self.n:
                i_swap = i_child1
            else:
                break

            if (i_swap > 0) and (data_arr[i_swap].val <= data_arr[i].val):
                swap_nodes(data_arr, i, i_swap)
                i = i_swap
            else:
                break
        
        return popped_element


######################################################################
# Ball Tree class

cdef struct NodeData_t:
    ITYPE_t idx_start
    ITYPE_t idx_end
    int is_leaf
    DTYPE_t radius

# use a dummy variable to determine the python data type
cdef NodeData_t dummy
cdef NodeData_t[:] dummy_view = <NodeData_t[:1]> &dummy
NodeData = np.asarray(dummy_view).dtype


cdef class BallTree:
    cdef readonly DTYPE_t[:, ::1] data
    cdef public ITYPE_t[::1] idx_array
    cdef public NodeData_t[::1] node_data
    cdef public DTYPE_t[:, ::1] centroids

    cdef ITYPE_t leaf_size
    cdef ITYPE_t n_levels
    cdef ITYPE_t n_nodes

    cdef DistanceMetric dm
    cdef int euclidean

    # variables to keep track of building & querying stats
    cdef int n_trims
    cdef int n_leaves
    cdef int n_splits
    cdef int n_calls

    # Use cinit to initialize all arrays to empty: this prevents errors
    # in rare cases where __init__ is not called
    def __cinit__(self):
        self.data = np.empty((0, 1), dtype=DTYPE, order='C')
        self.idx_array = np.empty(0, dtype=ITYPE, order='C')
        self.node_data = np.empty(0, dtype=NodeData, order='C')
        self.centroids = np.empty((0, 1), dtype=DTYPE)

        self.leaf_size = 0
        self.n_levels = 0
        self.n_nodes = 0
        self.n_calls = 0
        self.euclidean = False

    def __init__(self, DTYPE_t[:, ::1] data,
                 leaf_size=20, metric='euclidean', **kwargs):
        self.data = data
        self.leaf_size = leaf_size
        self.dm = DistanceMetric.get_metric(metric, **kwargs)
        self.euclidean = (self.dm.__class__.__name__ == 'EuclideanDistance')

        # validate data
        if self.data.size == 0:
            raise ValueError("X is an empty array")

        if leaf_size < 1:
            raise ValueError("leaf_size must be greater than or equal to 1")
        
        n_samples = self.data.shape[0]
        n_features = self.data.shape[1]

        # determine number of levels in the tree, and from this
        # the number of nodes in the tree.  This results in leaf nodes
        # with numbers of points betweeen leaf_size and 2 * leaf_size
        self.n_levels = np.log2(fmax(1, (n_samples - 1) / self.leaf_size)) + 1
        self.n_nodes = (2 ** self.n_levels) - 1

        # allocate arrays for storage
        self.idx_array = np.arange(n_samples, dtype=ITYPE)
        self.node_data = np.zeros(self.n_nodes, dtype=NodeData)

        # Allocate tree-specific data from TreeBase
        self.allocate_data(self.n_nodes, n_features)        
        self._recursive_build(0, 0, n_samples)

    def get_tree_stats(self):
        return (self.n_trims, self.n_leaves, self.n_splits)

    def reset_n_calls(self):
        self.n_calls = 0

    def get_n_calls(self):
        return self.n_calls

    cdef inline DTYPE_t dist(self, DTYPE_t* x1, DTYPE_t* x2,
                             ITYPE_t size):
        self.n_calls += 1
        if self.euclidean:
            return euclidean_dist(x1, x2, size)
        else:
            return self.dm.dist(x1, x2, size)

    cdef inline DTYPE_t rdist(self, DTYPE_t* x1, DTYPE_t* x2,
                              ITYPE_t size):
        self.n_calls += 1
        if self.euclidean:
            return euclidean_rdist(x1, x2, size)
        else:
            return self.dm.rdist(x1, x2, size)

    cdef void _recursive_build(self, ITYPE_t i_node,
                               ITYPE_t idx_start, ITYPE_t idx_end):
        cdef ITYPE_t imax
        cdef ITYPE_t n_features = self.data.shape[1]
        cdef ITYPE_t n_points = idx_end - idx_start
        cdef ITYPE_t n_mid = n_points / 2
        cdef ITYPE_t* idx_array = &self.idx_array[idx_start]
        cdef DTYPE_t* data = &self.data[0, 0]

        # initialize node data
        self.init_node(i_node, idx_start, idx_end)

        if 2 * i_node + 1 >= self.n_nodes:
            self.node_data[i_node].is_leaf = True
            if idx_end - idx_start > 2 * self.leaf_size:
                # this shouldn't happen if our memory allocation is correct
                # we'll proactively prevent memory errors, but raise a
                # warning saying we're doing so.
                import warnings
                warnings.warn("Internal: memory layout is flawed: "
                              "not enough nodes allocated")

        elif idx_end - idx_start < 2:
            # again, this shouldn't happen if our memory allocation
            # is correct.  Raise a warning.
            import warnings
            warnings.warn("Internal: memory layout is flawed: "
                          "too many nodes allocated")
            self.node_data[i_node].is_leaf = True

        else:
            # split node and recursively construct child nodes.
            self.node_data[i_node].is_leaf = False
            i_max = find_split_dim(data, idx_array,
                                   n_features, n_points)
            partition_indices(data, idx_array, i_max, n_mid,
                              n_features, n_points)
            self._recursive_build(2 * i_node + 1,
                                  idx_start, idx_start + n_mid)
            self._recursive_build(2 * i_node + 2,
                                  idx_start + n_mid, idx_end)

    def query(self, X, k=1, return_distance=True,
              dualtree=False, breadth_first=False):
        """
        query(X, k=1, return_distance=True)

        query the Ball Tree for the k nearest neighbors

        Parameters
        ----------
        X : array-like, last dimension same as that of self.data
            An array of points to query
        k : integer  (default = 1)
            The number of nearest neighbors to return
        return_distance : boolean (default = True)
            if True, return a tuple (d,i)
            if False, return array i

        Returns
        -------
        i    : if return_distance == False
        (d, i) : if return_distance == True

        d : array of doubles - shape: x.shape[:-1] + (k,)
            each entry gives the sorted list of distances to the
            neighbors of the corresponding point

        i : array of integers - shape: x.shape[:-1] + (k,)
            each entry gives the sorted list of indices of
            neighbors of the corresponding point

        Examples
        --------
        Query for k-nearest neighbors

            # >>> import numpy as np
            # >>> np.random.seed(0)
            # >>> X = np.random.random((10,3))  # 10 points in 3 dimensions
            # >>> ball_tree = BallTree(X, leaf_size=2)
            # >>> dist, ind = ball_tree.quer
            # >>> print ind  # indices of 3 closest neighbors
            # [0 3 1]
            # >>> print dist  # distances to 3 closest neighbors
            # [ 0.          0.19662693  0.29473397]
        """
        # XXX: if dualtree, allow X to be a pre-built tree.
        X = np.atleast_1d(np.asarray(X, dtype=DTYPE, order='C'))

        if X.shape[-1] != self.data.shape[1]:
            raise ValueError("query data dimension must "
                             "match training data dimension")

        if self.data.shape[0] < k:
            raise ValueError("k must be less than or equal "
                             "to the number of training points")

        # flatten X, and save original shape information
        cdef DTYPE_t[:, ::1] Xarr = X.reshape((-1, self.data.shape[1]))
        cdef DTYPE_t reduced_dist_LB
        cdef ITYPE_t i
        cdef DTYPE_t* pt

        # initialize heap for neighbors
        cdef NeighborsHeap heap = NeighborsHeap(Xarr.shape[0], k)

        # node heap for breadth-first queries
        cdef NodeHeap nodeheap
        if breadth_first:
            nodeheap = NodeHeap(self.data.shape[0] // self.leaf_size)

        # bounds is needed for the dual tree algorithm
        cdef DTYPE_t[::1] bounds

        self.n_trims = 0
        self.n_leaves = 0
        self.n_splits = 0

        if dualtree:
            other = self.__class__(Xarr, metric=self.dm,
                                   leaf_size=self.leaf_size)
            if breadth_first:
                self._query_dual_breadthfirst(other, heap, nodeheap)
            else:
                reduced_dist_LB = self.min_rdist_dual(0, other, 0)
                bounds = np.inf + np.zeros(other.node_data.shape[0])
                self._query_dual_depthfirst(0, other, 0, bounds,
                                            heap, reduced_dist_LB)

        else:
            pt = &Xarr[0, 0]
            if breadth_first:
                for i in range(Xarr.shape[0]):
                    self._query_one_breadthfirst(pt, i, heap, nodeheap)
                    pt += Xarr.shape[1]
            else:
                for i in range(Xarr.shape[0]):
                    reduced_dist_LB = self.min_rdist(0, pt)
                    self._query_one_depthfirst(0, pt, i, heap,
                                               reduced_dist_LB)
                    pt += Xarr.shape[1]

        distances, indices = heap.get_arrays(sort=True)
        distances = self.dm.rdist_to_dist_arr(distances)

        # deflatten results
        if return_distance:
            return (distances.reshape(X.shape[:-1] + (k,)),
                    indices.reshape(X.shape[:-1] + (k,)))
        else:
            return indices.reshape(X.shape[:-1] + (k,))


    def query_radius(self, X, r, return_distance=False,
                     int count_only=False, int sort_results=False):
        """
        query_radius(self, X, r, return_distance=False,
                     count_only = False, sort_results=False):

        query the Ball Tree for neighbors within a ball of size r

        Parameters
        ----------
        X : array-like
            An array of points to query.  Last dimension should match dimension
            of training data.
        r : distance within which neighbors are returned
            r can be a single value, or an array of values of shape
            x.shape[:-1] if different radii are desired for each point.
        return_distance : boolean (default = False)
            if True,  return distances to neighbors of each point
            if False, return only neighbors
            Note that unlike BallTree.query(), setting return_distance=True
            adds to the computation time.  Not all distances need to be
            calculated explicitly for return_distance=False.  Results are
            not sorted by default: see ``sort_results`` keyword.
        count_only : boolean (default = False)
            if True,  return only the count of points within distance r
            if False, return the indices of all points within distance r
            If return_distance==True, setting count_only=True will
            result in an error.
        sort_results : boolean (default = False)
            if True, the distances and indices will be sorted before being
            returned.  If False, the results will not be sorted.  If
            return_distance == False, setting sort_results = True will
            result in an error.

        Returns
        -------
        count       : if count_only == True
        ind         : if count_only == False and return_distance == False
        (ind, dist) : if count_only == False and return_distance == True

        count : array of integers, shape = X.shape[:-1]
            each entry gives the number of neighbors within
            a distance r of the corresponding point.

        ind : array of objects, shape = X.shape[:-1]
            each element is a numpy integer array listing the indices of
            neighbors of the corresponding point.  Note that unlike
            the results of BallTree.query(), the returned neighbors
            are not sorted by distance

        dist : array of objects, shape = X.shape[:-1]
            each element is a numpy double array
            listing the distances corresponding to indices in i.

        Examples
        --------
        Query for neighbors in a given radius

        # >>> import numpy as np
        # >>> np.random.seed(0)
        # >>> X = np.random.random((10,3))  # 10 points in 3 dimensions
        # >>> ball_tree = BallTree(X, leaf_size=2)
        # >>> print ball_tree.query_radius(X[0], r=0.3, count_only=True)
        # 3
        # >>> ind = ball_tree.query_radius(X[0], r=0.3)
        # >>> print ind  # indices of neighbors within distance 0.3
        # [3 0 1]
        """
        if count_only and return_distance:
            raise ValueError("count_only and return_distance "
                             "cannot both be true")

        if sort_results and not return_distance:
            raise ValueError("return_distance must be True "
                             "if sort_results is True")

        cdef ITYPE_t i, count_i = 0
        cdef ITYPE_t n_features = self.data.shape[1]
        cdef DTYPE_t[::1] dist_arr_i
        cdef ITYPE_t[::1] idx_arr_i, count_arr
        cdef DTYPE_t* pt

        # validate X and prepare for query
        X = np.atleast_2d(np.asarray(X, dtype=DTYPE, order='C'))

        if X.shape[-1] != self.data.shape[1]:
            raise ValueError("query data dimension must "
                             "match training data dimension")

        cdef DTYPE_t[:, ::1] Xarr = X.reshape((-1, self.data.shape[1]))

        # prepare r for query
        r = np.asarray(r, dtype=DTYPE, order='C')
        r = np.atleast_1d(r)
        if r.shape == (1,):
            r = r[0] + np.zeros(X.shape[:-1], dtype=DTYPE)
        else:
            if r.shape != X.shape[:-1]:
                raise ValueError("r must be broadcastable to X.shape")

        cdef DTYPE_t[::1] rarr = r.reshape(-1)

        # prepare variables for iteration
        if not count_only:
            indices = np.zeros(Xarr.shape[0], dtype='object')
            if return_distance:
                distances = np.zeros(Xarr.shape[0], dtype='object')

        idx_arr_i = np.zeros(self.data.shape[0], dtype=ITYPE)
        dist_arr_i = np.zeros(self.data.shape[0], dtype=DTYPE)
        count_arr = np.zeros(Xarr.shape[0], dtype=ITYPE)

        pt = &Xarr[0, 0]
        for i in range(Xarr.shape[0]):
            count_arr[i] = self._query_radius_one(0, pt, rarr[i],
                                                  &idx_arr_i[0],
                                                  &dist_arr_i[0],
                                                  0, count_only,
                                                  return_distance)
            pt += n_features

            if count_only:
                pass
            else:
                if sort_results:
                    _simultaneous_sort(&dist_arr_i[0], &idx_arr_i[0],
                                       count_arr[i])
 
                indices[i] = np.array(idx_arr_i[:count_arr[i]],
                                      copy=True)
                if return_distance:
                    distances[i] = np.array(dist_arr_i[:count_arr[i]],
                                            copy=True)

        # deflatten results
        if count_only:
            return np.asarray(count_arr).reshape(X.shape[:-1])
        elif return_distance:
            return (indices.reshape(X.shape[:-1]),
                    distances.reshape(X.shape[:-1]))
        else:
            return indices.reshape(X.shape[:-1])

    def kernel_density(BallTree self, X, h, kernel='gaussian',
                       atol=0, dualtree=False):
        """
        kernel_density(self, X, h, kernel='gaussian', atol=0, rtol=0,
                       dualtree=False)

        Compute the kernel density estimate of X with the given kernel.

        Parameters
        ----------
        X : array_like
            An array of points to query.  Last dimension should match dimension
            of training data.
        h : float
            the bandwidth of the kernel
        kernel : string
            specify the kernel to use.  Options are
            - 'gaussian'
            - 'tophat'
            - 'epanechnikov'
            - 'exponential'
            - 'linear'
            - 'cosine'
            Default is kernel = 'gaussian'
        atol : float
            specify the desired absolute tolerance of the result.
            Default is 0.
        dualtree : boolean
            if True, use the dual tree formalism.  This can be faster for
            large N.  Default is False
        """
        cdef DTYPE_t h_c = h
        cdef DTYPE_t atol_c = atol
        cdef ITYPE_t n_features = self.data.shape[1]
        cdef KernelType kernel_c

        # validate kernel
        if kernel == 'gaussian':
            kernel_c = GAUSSIAN_KERNEL
        elif kernel == 'tophat':
            kernel_c = TOPHAT_KERNEL
        elif kernel == 'epanechnikov':
            kernel_c = EPANECHNIKOV_KERNEL
        elif kernel == 'exponential':
            kernel_c = EXPONENTIAL_KERNEL
        elif kernel == 'linear':
            kernel_c = LINEAR_KERNEL
        elif kernel == 'cosine':
            kernel_c = COSINE_KERNEL
        else:
            raise ValueError("kernel = '%s' not recognized" % kernel)

        # validate X and prepare for query
        X = np.atleast_1d(np.asarray(X, dtype=DTYPE, order='C'))

        if X.shape[-1] != self.data.shape[1]:
            raise ValueError("query data dimension must "
                             "match training data dimension")
        cdef DTYPE_t[:, ::1] Xarr = X.reshape((-1, self.data.shape[1]))
        
        cdef DTYPE_t[::1] density = np.zeros(Xarr.shape[0], dtype=DTYPE)

        cdef DTYPE_t* pt = &Xarr[0, 0]

        # scale error to an error per point
        atol_c /= self.data.shape[0]

        if dualtree:
            other = self.__class__(Xarr, metric=self.dm,
                                   leaf_size=self.leaf_size)
            self._kernel_density_dual(0, other, 0, kernel_c,
                                      h_c, atol_c, &density[0])
        else:
            for i in range(Xarr.shape[0]):
                density[i] = self._kernel_density_one(0, pt, kernel_c,
                                                      h_c, atol_c, 0)
                pt += n_features

        # normalize the results
        cdef DTYPE_t knorm = kernel_norm(h_c, kernel_c)
        for i in range(density.shape[0]):
            density[i] *= knorm

        return np.asarray(density).reshape(X.shape[:-1])

    cdef void _query_one_depthfirst(BallTree self, ITYPE_t i_node,
                                    DTYPE_t* pt, ITYPE_t i_pt,
                                    NeighborsHeap heap,
                                    DTYPE_t reduced_dist_LB):
        cdef NodeData_t node_info = self.node_data[i_node]

        cdef DTYPE_t dist_pt, reduced_dist_LB_1, reduced_dist_LB_2
        cdef ITYPE_t i, i1, i2
        
        cdef DTYPE_t* data = &self.data[0, 0]

        #------------------------------------------------------------
        # Case 1: query point is outside node radius:
        #         trim it from the query
        if reduced_dist_LB > heap.largest(i_pt):
            self.n_trims += 1

        #------------------------------------------------------------
        # Case 2: this is a leaf node.  Update set of nearby points
        elif node_info.is_leaf:
            self.n_leaves += 1
            for i in range(node_info.idx_start, node_info.idx_end):
                dist_pt = self.rdist(pt,
                                     &self.data[self.idx_array[i], 0],
                                     self.data.shape[1])
                if dist_pt < heap.largest(i_pt):
                    heap.push(i_pt, dist_pt, self.idx_array[i])

        #------------------------------------------------------------
        # Case 3: Node is not a leaf.  Recursively query subnodes
        #         starting with the closest
        else:
            self.n_splits += 1
            i1 = 2 * i_node + 1
            i2 = i1 + 1
            reduced_dist_LB_1 = self.min_rdist(i1, pt)
            reduced_dist_LB_2 = self.min_rdist(i2, pt)

            # recursively query subnodes
            if reduced_dist_LB_1 <= reduced_dist_LB_2:
                self._query_one_depthfirst(i1, pt, i_pt, heap,
                                           reduced_dist_LB_1)
                self._query_one_depthfirst(i2, pt, i_pt, heap,
                                           reduced_dist_LB_2)
            else:
                self._query_one_depthfirst(i2, pt, i_pt, heap,
                                           reduced_dist_LB_2)
                self._query_one_depthfirst(i1, pt, i_pt, heap,
                                           reduced_dist_LB_1)

    cdef void _query_one_breadthfirst(BallTree self, DTYPE_t* pt,
                                      ITYPE_t i_pt,
                                      NeighborsHeap heap,
                                      NodeHeap nodeheap):
        cdef ITYPE_t i, i_node
        cdef DTYPE_t dist_pt, reduced_dist_LB
        cdef NodeData_t* node_data = &self.node_data[0]
        cdef DTYPE_t* data = &self.data[0, 0]

        # Set up the node heap and push the head node onto it
        cdef NodeHeapData_t nodeheap_item
        nodeheap_item.val = self.min_rdist(0, pt)
        nodeheap_item.i1 = 0
        nodeheap.push(nodeheap_item)

        while nodeheap.n > 0:
            nodeheap_item = nodeheap.pop()
            reduced_dist_LB = nodeheap_item.val
            i_node = nodeheap_item.i1
            node_info = node_data[i_node]

            #------------------------------------------------------------
            # Case 1: query point is outside node radius:
            #         trim it from the query
            if reduced_dist_LB > heap.largest(i_pt):
                self.n_trims += 1

            #------------------------------------------------------------
            # Case 2: this is a leaf node.  Update set of nearby points
            elif node_data[i_node].is_leaf:
                self.n_leaves += 1
                for i in range(node_data[i_node].idx_start,
                               node_data[i_node].idx_end):
                    dist_pt = self.rdist(pt,
                                         &self.data[self.idx_array[i], 0],
                                         self.data.shape[1])
                    if dist_pt < heap.largest(i_pt):
                        heap.push(i_pt, dist_pt, self.idx_array[i])

            #------------------------------------------------------------
            # Case 3: Node is not a leaf.  Add subnodes to the node heap
            else:
                self.n_splits += 1
                for i in range(2 * i_node + 1, 2 * i_node + 3):
                    nodeheap_item.i1 = i
                    nodeheap_item.val = self.min_rdist(i, pt)
                    nodeheap.push(nodeheap_item)

    cdef void _query_dual_depthfirst(BallTree self, ITYPE_t i_node1,
                                     BallTree other, ITYPE_t i_node2,
                                     DTYPE_t[::1] bounds,
                                     NeighborsHeap heap,
                                     DTYPE_t reduced_dist_LB):
        # note that the array `bounds` is maintained such that
        # bounds[i] is the largest distance among any of the
        # current neighbors in node i of the other tree.
        cdef NodeData_t node_info1 = self.node_data[i_node1]
        cdef NodeData_t node_info2 = other.node_data[i_node2]

        cdef DTYPE_t* data1 = &self.data[0, 0]
        cdef DTYPE_t* data2 = &other.data[0, 0]
        cdef ITYPE_t n_features = self.data.shape[1]

        cdef DTYPE_t bound_max, dist_pt, reduced_dist_LB1, reduced_dist_LB2
        cdef ITYPE_t i1, i2, i_pt, i_parent

        #------------------------------------------------------------
        # Case 1: nodes are further apart than the current bound:
        #         trim both from the query
        if reduced_dist_LB > bounds[i_node2]:
            pass

        #------------------------------------------------------------
        # Case 2: both nodes are leaves:
        #         do a brute-force search comparing all pairs
        elif node_info1.is_leaf and node_info2.is_leaf:
            bounds[i_node2] = 0

            for i2 in range(node_info2.idx_start, node_info2.idx_end):
                i_pt = other.idx_array[i2]
                
                if heap.largest(i_pt) <= reduced_dist_LB:
                    continue

                for i1 in range(node_info1.idx_start, node_info1.idx_end):
                    dist_pt = self.rdist(
                        data1 + n_features * self.idx_array[i1],
                        data2 + n_features * i_pt,
                        n_features)
                    if dist_pt < heap.largest(i_pt):
                        heap.push(i_pt, dist_pt, self.idx_array[i1])
                
                # keep track of node bound
                bounds[i_node2] = fmax(bounds[i_node2],
                                       heap.largest(i_pt))

            # update bounds up the tree
            while i_node2 > 0:
                i_parent = (i_node2 - 1) // 2
                bound_max = fmax(bounds[2 * i_parent + 1],
                                 bounds[2 * i_parent + 2])
                if bound_max < bounds[i_parent]:
                    bounds[i_parent] = bound_max
                    i_node2 = i_parent
                else:
                    break
            
        #------------------------------------------------------------
        # Case 3a: node 1 is a leaf: split node 2 and recursively
        #          query, starting with the nearest node
        elif node_info1.is_leaf:
            reduced_dist_LB1 = self.min_rdist_dual(i_node1,
                                                   other, 2 * i_node2 + 1)
            reduced_dist_LB2 = self.min_rdist_dual(i_node1,
                                                   other, 2 * i_node2 + 2)

            if reduced_dist_LB1 < reduced_dist_LB2:
                self._query_dual_depthfirst(i_node1, other, 2 * i_node2 + 1,
                                            bounds, heap, reduced_dist_LB1)
                self._query_dual_depthfirst(i_node1, other, 2 * i_node2 + 2,
                                            bounds, heap, reduced_dist_LB2)
            else:
                self._query_dual_depthfirst(i_node1, other, 2 * i_node2 + 2,
                                            bounds, heap, reduced_dist_LB2)
                self._query_dual_depthfirst(i_node1, other, 2 * i_node2 + 1,
                                            bounds, heap, reduced_dist_LB1)
            
        #------------------------------------------------------------
        # Case 3b: node 2 is a leaf: split node 1 and recursively
        #          query, starting with the nearest node
        elif node_info2.is_leaf:
            reduced_dist_LB1 = self.min_rdist_dual(2 * i_node1 + 1,
                                                   other, i_node2)
            reduced_dist_LB2 = self.min_rdist_dual(2 * i_node1 + 2,
                                                   other, i_node2)

            if reduced_dist_LB1 < reduced_dist_LB2:
                self._query_dual_depthfirst(2 * i_node1 + 1, other, i_node2,
                                            bounds, heap, reduced_dist_LB1)
                self._query_dual_depthfirst(2 * i_node1 + 2, other, i_node2,
                                            bounds, heap, reduced_dist_LB2)
            else:
                self._query_dual_depthfirst(2 * i_node1 + 2, other, i_node2,
                                            bounds, heap, reduced_dist_LB2)
                self._query_dual_depthfirst(2 * i_node1 + 1, other, i_node2,
                                            bounds, heap, reduced_dist_LB1)
        
        #------------------------------------------------------------
        # Case 4: neither node is a leaf:
        #         split both and recursively query all four pairs
        else:
            reduced_dist_LB1 = self.min_rdist_dual(2 * i_node1 + 1,
                                                   other, 2 * i_node2 + 1)
            reduced_dist_LB2 = self.min_rdist_dual(2 * i_node1 + 2,
                                                   other, 2 * i_node2 + 1)

            if reduced_dist_LB1 < reduced_dist_LB2:
                self._query_dual_depthfirst(2 * i_node1 + 1,
                                            other, 2 * i_node2 + 1,
                                            bounds, heap, reduced_dist_LB1)
                self._query_dual_depthfirst(2 * i_node1 + 2,
                                            other, 2 * i_node2 + 1,
                                            bounds, heap, reduced_dist_LB2)
            else:
                self._query_dual_depthfirst(2 * i_node1 + 2,
                                            other, 2 * i_node2 + 1,
                                            bounds, heap, reduced_dist_LB2)
                self._query_dual_depthfirst(2 * i_node1 + 1,
                                            other, 2 * i_node2 + 1,
                                            bounds, heap, reduced_dist_LB1)

            reduced_dist_LB1 = self.min_rdist_dual(2 * i_node1 + 1,
                                                   other, 2 * i_node2 + 2)
            reduced_dist_LB2 = self.min_rdist_dual(2 * i_node1 + 2,
                                                   other, 2 * i_node2 + 2)
            if reduced_dist_LB1 < reduced_dist_LB2:
                self._query_dual_depthfirst(2 * i_node1 + 1,
                                            other, 2 * i_node2 + 2,
                                            bounds, heap, reduced_dist_LB1)
                self._query_dual_depthfirst(2 * i_node1 + 2,
                                            other, 2 * i_node2 + 2,
                                            bounds, heap, reduced_dist_LB2)
            else:
                self._query_dual_depthfirst(2 * i_node1 + 2,
                                            other, 2 * i_node2 + 2,
                                            bounds, heap, reduced_dist_LB2)
                self._query_dual_depthfirst(2 * i_node1 + 1,
                                            other, 2 * i_node2 + 2,
                                            bounds, heap, reduced_dist_LB1)

    cdef void _query_dual_breadthfirst(BallTree self, BallTree other,
                                       NeighborsHeap heap, NodeHeap nodeheap):
        cdef ITYPE_t i, i1, i2, i_node1, i_node2, i_pt
        cdef DTYPE_t dist_pt, reduced_dist_LB
        cdef DTYPE_t[::1] bounds = np.inf + np.zeros(other.node_data.shape[0])
        cdef NodeData_t* node_data1 = &self.node_data[0]
        cdef NodeData_t* node_data2 = &other.node_data[0]
        cdef NodeData_t node_info1, node_info2
        cdef DTYPE_t* data1 = &self.data[0, 0]
        cdef DTYPE_t* data2 = &other.data[0, 0]
        cdef ITYPE_t n_features = self.data.shape[1]

        # Set up the node heap and push the head nodes onto it
        cdef NodeHeapData_t nodeheap_item
        nodeheap_item.val = self.min_rdist_dual(0, other, 0)
        nodeheap_item.i1 = 0
        nodeheap_item.i2 = 0
        nodeheap.push(nodeheap_item)

        while nodeheap.n > 0:
            nodeheap_item = nodeheap.pop()
            reduced_dist_LB = nodeheap_item.val
            i_node1 = nodeheap_item.i1
            i_node2 = nodeheap_item.i2
            
            node_info1 = node_data1[i_node1]
            node_info2 = node_data2[i_node2]

            #------------------------------------------------------------
            # Case 1: nodes are further apart than the current bound:
            #         trim both from the query
            if reduced_dist_LB > bounds[i_node2]:
                pass

            #------------------------------------------------------------
            # Case 2: both nodes are leaves:
            #         do a brute-force search comparing all pairs
            elif node_info1.is_leaf and node_info2.is_leaf:
                bounds[i_node2] = -1

                for i2 in range(node_info2.idx_start, node_info2.idx_end):
                    i_pt = other.idx_array[i2]
                
                    if heap.largest(i_pt) <= reduced_dist_LB:
                        continue

                    for i1 in range(node_info1.idx_start, node_info1.idx_end):
                        dist_pt = self.rdist(
                            data1 + n_features * self.idx_array[i1],
                            data2 + n_features * i_pt,
                            n_features)
                        if dist_pt < heap.largest(i_pt):
                            heap.push(i_pt, dist_pt, self.idx_array[i1])
                
                    # keep track of node bound
                    bounds[i_node2] = fmax(bounds[i_node2],
                                           heap.largest(i_pt))
            
            #------------------------------------------------------------
            # Case 3a: node 1 is a leaf: split node 2 and recursively
            #          query, starting with the nearest node
            elif node_info1.is_leaf:
                nodeheap_item.i1 = i_node1
                for i2 in range(2 * i_node2 + 1, 2 * i_node2 + 3):
                    nodeheap_item.i2 = i2
                    nodeheap_item.val = self.min_rdist_dual(i_node1, other, i2)
                    nodeheap.push(nodeheap_item)
            
            #------------------------------------------------------------
            # Case 3b: node 2 is a leaf: split node 1 and recursively
            #          query, starting with the nearest node
            elif node_info2.is_leaf:
                nodeheap_item.i2 = i_node2
                for i1 in range(2 * i_node1 + 1, 2 * i_node1 + 3):
                    nodeheap_item.i1 = i1
                    nodeheap_item.val = self.min_rdist_dual(i1, other, i_node2)
                    nodeheap.push(nodeheap_item)
        
            #------------------------------------------------------------
            # Case 4: neither node is a leaf:
            #         split both and recursively query all four pairs
            else:
                for i1 in range(2 * i_node1 + 1, 2 * i_node1 + 3):
                    for i2 in range(2 * i_node2 + 1, 2 * i_node2 + 3):
                        nodeheap_item.i1 = i1
                        nodeheap_item.i2 = i2
                        nodeheap_item.val = self.min_rdist_dual(i1, other, i2)
                        nodeheap.push(nodeheap_item)

    cdef ITYPE_t _query_radius_one(BallTree self,
                                   ITYPE_t i_node,
                                   DTYPE_t* pt, DTYPE_t r,
                                   ITYPE_t* indices,
                                   DTYPE_t* distances,
                                   ITYPE_t count,
                                   int count_only,
                                   int return_distance):
        cdef DTYPE_t* data = &self.data[0, 0]
        cdef ITYPE_t* idx_array = &self.idx_array[0]
        cdef ITYPE_t n_features = self.data.shape[1]
        cdef NodeData_t node_info = self.node_data[i_node]

        cdef ITYPE_t i
        cdef DTYPE_t reduced_r

        # XXX: for efficiency, calculate dist_LB and dist_UB at the same time
        cdef DTYPE_t dist_pt, dist_LB, dist_UB
        dist_LB = self.min_dist(i_node, pt)
        dist_UB = self.max_dist(i_node, pt)

        #------------------------------------------------------------
        # Case 1: all node points are outside distance r.
        #         prune this branch.
        if dist_LB > r:
            pass

        #------------------------------------------------------------
        # Case 2: all node points are within distance r
        #         add all points to neighbors
        elif dist_UB <= r:
            if count_only:
                count += (node_info.idx_end - node_info.idx_start)
            else:
                for i in range(node_info.idx_start, node_info.idx_end):
                    if (count < 0) or (count >= self.data.shape[0]):
                        raise ValueError("Fatal: count too big: "
                                         "this should never happen")
                    indices[count] = idx_array[i]
                    if return_distance:
                        distances[count] = self.dist(pt, (data + n_features
                                                          * idx_array[i]),
                                                     n_features)
                    count += 1

        #------------------------------------------------------------
        # Case 3: this is a leaf node.  Go through all points to
        #         determine if they fall within radius
        elif node_info.is_leaf:
            reduced_r = self.dm.dist_to_rdist(r)

            for i in range(node_info.idx_start, node_info.idx_end):
                dist_pt = self.rdist(pt, (data + n_features * idx_array[i]),
                                     n_features)
                if dist_pt <= reduced_r:
                    if (count < 0) or (count >= self.data.shape[0]):
                        raise ValueError("Fatal: count out of range. "
                                         "This should never happen.")
                    if count_only:
                        pass
                    else:
                        indices[count] = idx_array[i]
                        if return_distance:
                            distances[count] = self.dm.rdist_to_dist(dist_pt)
                    count += 1

        #------------------------------------------------------------
        # Case 4: Node is not a leaf.  Recursively query subnodes
        else:
            count = self._query_radius_one(2 * i_node + 1, pt, r,
                                           indices, distances, count,
                                           count_only, return_distance)
            count = self._query_radius_one(2 * i_node + 2, pt, r,
                                           indices, distances, count,
                                           count_only, return_distance)

        return count

    cdef DTYPE_t _kernel_density_one(self, ITYPE_t i_node, DTYPE_t* pt,
                                     KernelType kernel,
                                     DTYPE_t h, DTYPE_t atol, DTYPE_t dens):
        cdef ITYPE_t i

        # note that atol here is absolute tolerance *per point*
        cdef DTYPE_t* data = &self.data[0, 0]
        cdef ITYPE_t* idx_array = &self.idx_array[0]
        cdef ITYPE_t n_features = self.data.shape[1]

        cdef NodeData_t node_info = self.node_data[i_node]
        cdef DTYPE_t dist_pt

        # XXX: for efficiency, calculate dist_LB and dist_UB at the same time
        cdef DTYPE_t dist_LB = self.min_dist(i_node, pt)
        cdef DTYPE_t dist_UB = self.max_dist(i_node, pt)

        cdef DTYPE_t dens_UB = compute_kernel(dist_LB, h, kernel)
        cdef DTYPE_t dens_LB = compute_kernel(dist_UB, h, kernel)
        cdef DTYPE_t knorm = kernel_norm(h, kernel)

        #------------------------------------------------------------
        # Case 1: points are all far enough that contribution is
        #         zero to machine precision.  Trim the node
        if knorm * dens_UB <= atol:
            dens += dens_UB * (node_info.idx_end - node_info.idx_start)
        
        #------------------------------------------------------------
        # Case 2: points are all close enough that the contribution
        #         is maximal.  Add all points
        elif knorm * dens_LB >= knorm - atol:
            dens += dens_LB * (node_info.idx_end - node_info.idx_start)

        #------------------------------------------------------------
        # Case 3: this is a leaf node.  Add all points
        elif node_info.is_leaf:
            for i in range(node_info.idx_start, node_info.idx_end):
                dist_pt = self.dist(pt, (data + n_features * idx_array[i]),
                                    n_features)
                dens += compute_kernel(dist_pt, h, kernel)

        #------------------------------------------------------------
        # Case 4: node needs to be split: recursively query subnodes
        else:
            dens = self._kernel_density_one(2 * i_node + 1, pt,
                                            kernel, h, atol, dens)
            dens = self._kernel_density_one(2 * i_node + 2, pt,
                                            kernel, h, atol, dens)

        return dens

    cdef void _kernel_density_dual(BallTree self, ITYPE_t i_node1,
                                   BallTree other, ITYPE_t i_node2,
                                   KernelType kernel,
                                   DTYPE_t h, DTYPE_t atol,
                                   DTYPE_t* density):
        cdef ITYPE_t i1, i2

        # note that atol here is absolute tolerance *per point*
        cdef DTYPE_t* data1 = &self.data[0, 0]
        cdef DTYPE_t* data2 = &other.data[0, 0]

        cdef ITYPE_t* idx_array1 = &self.idx_array[0]
        cdef ITYPE_t* idx_array2 = &other.idx_array[0]

        cdef ITYPE_t n_features = self.data.shape[1]

        cdef NodeData_t node_info1 = self.node_data[i_node1]
        cdef NodeData_t node_info2 = other.node_data[i_node2]
        cdef DTYPE_t dist_pt

        # XXX: for efficiency, calculate dist_LB and dist_UB at the same time
        cdef DTYPE_t dist_LB = self.min_dist_dual(i_node1, other, i_node2)
        cdef DTYPE_t dist_UB = self.max_dist_dual(i_node1, other, i_node2)

        cdef DTYPE_t dens_UB = compute_kernel(dist_LB, h, kernel)
        cdef DTYPE_t dens_LB = compute_kernel(dist_UB, h, kernel)
        cdef DTYPE_t knorm = kernel_norm(h, kernel)

        #------------------------------------------------------------
        # Case 1: points are all far enough that contribution is zero
        #         to within the desired tolerance
        if knorm * dens_UB <= atol:
            dens_UB *= (node_info1.idx_end - node_info1.idx_start)
            for i2 in range(node_info2.idx_start, node_info2.idx_end):
                density[idx_array2[i2]] += dens_UB
        
        #------------------------------------------------------------
        # Case 2: points are all close enough that the contribution
        #         is maximal to within the desired tolerance
        elif knorm * dens_LB >= knorm - atol:
            dens_LB *= (node_info1.idx_end - node_info1.idx_start)
            for i2 in range(node_info2.idx_start, node_info2.idx_end):
                density[idx_array2[i2]] += dens_LB

        #------------------------------------------------------------
        # Case 3: both nodes are leaves: go through all pairs
        elif node_info1.is_leaf and node_info2.is_leaf:
            for i2 in range(node_info2.idx_start, node_info2.idx_end):
                for i1 in range(node_info1.idx_start, node_info1.idx_end):
                    dist_pt = self.dist(data1 + n_features * idx_array1[i1],
                                        data2 + n_features * idx_array2[i2],
                                        n_features)
                    density[idx_array2[i2]] += compute_kernel(dist_pt, h,
                                                              kernel)

        #------------------------------------------------------------
        # Case 4a: only one node is a leaf: split the other
        elif node_info1.is_leaf:
            for i2 in range(2 * i_node2 + 1, 2 * i_node2 + 3):
                self._kernel_density_dual(i_node1, other, i2, kernel,
                                          h, atol, density)

        elif node_info2.is_leaf:
            for i1 in range(2 * i_node1 + 1, 2 * i_node1 + 3):
                self._kernel_density_dual(i1, other, i_node2, kernel,
                                          h, atol, density)

        #------------------------------------------------------------
        # Case 4b: both nodes need to be split
        else:
            for i1 in range(2 * i_node1 + 1, 2 * i_node1 + 3):
                for i2 in range(2 * i_node2 + 1, 2 * i_node2 + 3):
                    self._kernel_density_dual(i1, other, i2, kernel,
                                              h, atol, density)

    #----------------------------------------------------------------------
    # The following methods can be changed to produce a different tree type
    def get_arrays(self):
        return map(np.asarray, (self.data, self.idx_array, self.node_data,
                                self.centroids))

    cdef void allocate_data(self, ITYPE_t n_nodes, ITYPE_t n_features):
        self.centroids = np.zeros((n_nodes, n_features), dtype=DTYPE)

    cdef void init_node(self, ITYPE_t i_node,
                        ITYPE_t idx_start, ITYPE_t idx_end):
        cdef ITYPE_t n_features = self.data.shape[1]
        cdef ITYPE_t n_points = idx_end - idx_start

        cdef ITYPE_t i, j
        cdef DTYPE_t radius
        cdef DTYPE_t *this_pt

        cdef ITYPE_t* idx_array = &self.idx_array[0]
        cdef DTYPE_t* data = &self.data[0, 0]
        cdef DTYPE_t* centroid = &self.centroids[i_node, 0]

        # determine Node centroid
        for j in range(n_features):
            centroid[j] = 0

        for i in range(idx_start, idx_end):
            this_pt = data + n_features * idx_array[i]
            for j from 0 <= j < n_features:
                centroid[j] += this_pt[j]

        for j in range(n_features):
            centroid[j] /= n_points

        # determine Node radius
        radius = 0
        for i in range(idx_start, idx_end):
            radius = fmax(radius,
                          self.rdist(centroid,
                                     data + n_features * idx_array[i],
                                     n_features))

        self.node_data[i_node].radius = self.dm.rdist_to_dist(radius)
        self.node_data[i_node].idx_start = idx_start
        self.node_data[i_node].idx_end = idx_end

    cdef inline DTYPE_t min_dist(self, ITYPE_t i_node, DTYPE_t* pt):
        cdef DTYPE_t dist_pt = self.dist(pt, &self.centroids[i_node, 0],
                                         self.data.shape[1])
        return fmax(0, dist_pt - self.node_data[i_node].radius)

    cdef inline DTYPE_t max_dist(self, ITYPE_t i_node, DTYPE_t* pt):
        cdef DTYPE_t dist_pt = self.dist(pt, &self.centroids[i_node, 0],
                                         self.data.shape[1])
        return dist_pt + self.node_data[i_node].radius

    cdef inline DTYPE_t min_rdist(self, ITYPE_t i_node, DTYPE_t* pt):
        if self.euclidean:
            return euclidean_dist_to_rdist(self.min_dist(i_node, pt))
        else:
            return self.dm.dist_to_rdist(self.min_dist(i_node, pt))

    cdef inline DTYPE_t max_rdist(self, ITYPE_t i_node, DTYPE_t* pt):
        if self.euclidean:
            return euclidean_dist_to_rdist(self.max_dist(i_node, pt))
        else:
            return self.dm.dist_to_rdist(self.max_dist(i_node, pt))

    cdef inline DTYPE_t min_dist_dual(self, ITYPE_t i_node1,
                                      BallTree other, ITYPE_t i_node2):
        cdef DTYPE_t dist_pt = self.dist(&other.centroids[i_node2, 0],
                                          &self.centroids[i_node1, 0],
                                          self.data.shape[1])
        return fmax(0, (dist_pt - self.node_data[i_node1].radius
                        - other.node_data[i_node2].radius))

    cdef inline DTYPE_t max_dist_dual(self, ITYPE_t i_node1,
                                      BallTree other, ITYPE_t i_node2):
        cdef DTYPE_t dist_pt = self.dist(&other.centroids[i_node2, 0],
                                          &self.centroids[i_node1, 0],
                                          self.data.shape[1])
        return (dist_pt + self.node_data[i_node1].radius
                + other.node_data[i_node2].radius)

    cdef inline DTYPE_t min_rdist_dual(self, ITYPE_t i_node1,
                                       BallTree other, ITYPE_t i_node2):
        if self.euclidean:
            return euclidean_dist_to_rdist(self.min_dist_dual(i_node1,
                                                              other, i_node2))
        else:
            return self.dm.dist_to_rdist(self.min_dist_dual(i_node1,
                                                            other, i_node2))

    cdef inline DTYPE_t max_rdist_dual(self, ITYPE_t i_node1,
                                       BallTree other, ITYPE_t i_node2):
        if self.euclidean:
            return euclidean_dist_to_rdist(self.max_dist_dual(i_node1,
                                                              other, i_node2))
        else:
            return self.dm.dist_to_rdist(self.max_dist_dual(i_node1,
                                                            other, i_node2))


######################################################################
# Python functions for benchmarking and testing
def euclidean_pairwise_inline(DTYPE_t[:, ::1] X, DTYPE_t[:, ::1] Y):
    D = euclidean_cdist(X, Y)
    return np.asarray(D)


def euclidean_pairwise_class(DTYPE_t[:, ::1] X, DTYPE_t[:, ::1] Y):
    cdef EuclideanDistance eucl_dist = EuclideanDistance()

    assert X.shape[1] == Y.shape[1]
    cdef DTYPE_t[:, ::1] D = np.zeros((X.shape[0], Y.shape[0]), dtype=DTYPE)

    for i1 in range(X.shape[0]):
        for i2 in range(Y.shape[0]):
            D[i1, i2] = eucl_dist.dist(&X[i1, 0], &Y[i2, 0], X.shape[1])
    return np.asarray(D)


def euclidean_pairwise_polymorphic(DTYPE_t[:, ::1] X, DTYPE_t[:, ::1] Y):
    cdef DistanceMetric eucl_dist = EuclideanDistance()

    assert X.shape[1] == Y.shape[1]
    cdef DTYPE_t[:, ::1] D = np.zeros((X.shape[0], Y.shape[0]), dtype=DTYPE)

    for i1 in range(X.shape[0]):
        for i2 in range(Y.shape[0]):
            D[i1, i2] = eucl_dist.dist(&X[i1, 0], &Y[i2, 0], X.shape[1])
    return np.asarray(D)


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

def nodeheap_sort(DTYPE_t[::1] vals):
    """In-place reverse sort of vals using NodeHeap"""
    cdef ITYPE_t[::1] indices = np.zeros(vals.shape[0], dtype=ITYPE)
    cdef DTYPE_t[::1] vals_sorted = np.zeros_like(vals)

    # use initial size 0 to check corner case
    cdef NodeHeap heap = NodeHeap(0)
    cdef NodeHeapData_t data
    cdef ITYPE_t i
    for i in range(vals.shape[0]):
        data.val = vals[i]
        data.i1 = i
        data.i2 = i + 1
        heap.push(data)

    for i in range(vals.shape[0]):
        data = heap.pop()
        vals_sorted[i] = data.val
        indices[i] = data.i1

    return np.asarray(vals_sorted), np.asarray(indices)

#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

# Ball Tree with memoryviews

cimport cython
cimport numpy as np
from libc.math cimport fmax, fmin, fabs, sqrt

import numpy as np
import warnings

######################################################################
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


######################################################################
# Inline distance functions
cdef inline DTYPE_t euclidean_dist(DTYPE_t[:, ::1] X1, ITYPE_t i1,
                                   DTYPE_t[:, ::1] X2, ITYPE_t i2):
    cdef DTYPE_t tmp, d=0
    for j in range(X1.shape[1]):
        tmp = X1[i1, j] - X2[i2, j]
        d += tmp * tmp
    return sqrt(d)


cdef inline DTYPE_t euclidean_rdist(DTYPE_t[:, ::1] X1, ITYPE_t i1,
                                    DTYPE_t[:, ::1] X2, ITYPE_t i2):
    cdef DTYPE_t tmp, d=0
    for j in range(X1.shape[1]):
        tmp = X1[i1, j] - X2[i2, j]
        d += tmp * tmp
    return d


cdef DTYPE_t[:, ::1] euclidean_cdist(DTYPE_t[:, ::1] X, DTYPE_t[:, ::1] Y):
    if X.shape[1] != Y.shape[1]:
        raise ValueError('X and Y must have the same second dimension')

    cdef DTYPE_t[:, ::1] D = np.zeros((X.shape[0], Y.shape[0]), dtype=DTYPE)

    for i1 in range(X.shape[0]):
        for i2 in range(Y.shape[0]):
            D[i1, i2] = euclidean_dist(X, i1, Y, i2)
    return D


######################################################################
# Distance Metric Classes

#------------------------------------------------------------
# DistanceMetric base class
cdef class DistanceMetric:
    @classmethod
    def get_metric(cls, metric, **kwargs):
        if metric in [None, 'euclidean', 'l2', EuclideanDistance]:
            return EuclideanDistance(**kwargs)
        else:
            raise ValueError('metric = "%s" not recognized' % str(metric))

    def __init__(self, **kwargs):
        if self.__class__ is DistanceMetric:
            raise NotImplementedError("DistanceMetric is an abstract class")

    cdef DTYPE_t dist(self, DTYPE_t[:, ::1] X1, ITYPE_t i1,
                      DTYPE_t[:, ::1] X2, ITYPE_t i2):
        return -999

    cdef DTYPE_t rdist(self, DTYPE_t[:, ::1] X1, ITYPE_t i1,
                       DTYPE_t[:, ::1] X2, ITYPE_t i2):
        return self.dist(X1, i1, X2, i2)

    cdef DTYPE_t[:, ::1] pdist(self, DTYPE_t[:, ::1] X):
        cdef ITYPE_t i1, i2
        cdef DTYPE_t[:, ::1] D = np.zeros((X.shape[0], X.shape[0]),
                                          dtype=DTYPE, order='C')
        for i1 in range(X.shape[0]):
            for i2 in range(i1, X.shape[0]):
                D[i1, i2] = self.dist(X, i1, X, i2)
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
                D[i1, i2] = self.dist(X, i1, Y, i2)
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
    cdef inline DTYPE_t dist(self, DTYPE_t[:, ::1] X1, ITYPE_t i1,
                             DTYPE_t[:, ::1] X2, ITYPE_t i2):
        cdef DTYPE_t tmp, d=0
        for j in range(X1.shape[1]):
            tmp = X1[i1, j] - X2[i2, j]
            d += tmp * tmp
        return sqrt(d)

    cdef inline DTYPE_t rdist(self, DTYPE_t[:, ::1] X1, ITYPE_t i1,
                              DTYPE_t[:, ::1] X2, ITYPE_t i2):
        cdef DTYPE_t tmp, d=0
        for j in range(X1.shape[1]):
            tmp = X1[i1, j] - X2[i2, j]
            d += tmp * tmp
        return d

    cdef inline DTYPE_t rdist_to_dist(self, DTYPE_t rdist):
        return sqrt(rdist)

    cdef inline DTYPE_t dist_to_rdist(self, DTYPE_t dist):
        return dist * dist

    def rdist_to_dist_arr(self, rdist):
        return np.sqrt(rdist)

    def dist_to_rdist_arr(self, dist):
        return dist ** 2


######################################################################
# Tree Utility Routines
cdef inline void swap1(DITYPE_t[:, ::1] arr, ITYPE_t row,
                      ITYPE_t i1, ITYPE_t i2):
    cdef DITYPE_t tmp = arr[row, i1]
    arr[row, i1] = arr[row, i2]
    arr[row, i2] = tmp

cdef inline void swap2(DITYPE_t[::1] arr, ITYPE_t i1, ITYPE_t i2):
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

    cdef inline DTYPE_t largest(self, ITYPE_t row):
        return self.distances[row, 0]

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

    cdef void _sort(self):
        cdef DTYPE_t[:, ::1] distances = self.distances
        cdef ITYPE_t[:, ::1] indices = self.indices
        cdef ITYPE_t row
        for row in range(distances.shape[0]):
            _simultaneous_sort(distances, indices, row,
                               0, distances.shape[1])

    def get_arrays(self, sort=True):
        if sort:
            self._sort()
        return map(np.asarray, (self.distances, self.indices))


#------------------------------------------------------------
# simultaneous_sort :
#  this is a recursive quicksort implementation which sorts
#  distances[row, lower:upper] and simultaneously performs
#  the same swaps on `indices`.
cdef void _simultaneous_sort(DTYPE_t[:, ::1] distances,
                             ITYPE_t[:, ::1] indices,
                             ITYPE_t row, ITYPE_t lower, ITYPE_t upper):
    cdef DTYPE_t pivot_val
    cdef ITYPE_t pivot_idx, store_idx, i

    if lower + 1 >= upper:
        return

    # determine new pivot
    pivot_idx = (lower + upper) / 2
    pivot_val = distances[row, pivot_idx]
    store_idx = lower
    swap1(distances, row, pivot_idx, upper - 1)
    swap1(indices, row, pivot_idx, upper - 1)
    for i in range(lower, upper - 1):
        if distances[row, i] < pivot_val:
            swap1(distances, row, i, store_idx)
            swap1(indices, row, i, store_idx)
            store_idx += 1
    swap1(distances, row, store_idx, upper - 1)
    swap1(indices, row, store_idx, upper - 1)
    pivot_idx = store_idx

    # recursively sort each side of the pivot
    if lower + 1 < pivot_idx:
        _simultaneous_sort(distances, indices, row, lower, pivot_idx)
    if pivot_idx + 2 < upper:
        _simultaneous_sort(distances, indices, row, pivot_idx + 1, upper)


#------------------------------------------------------------
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


#------------------------------------------------------------
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
cpdef int partition_indices(DTYPE_t[:, ::1] data,
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
                swap2(indices, i, midindex)
                midindex += 1
        swap2(indices, midindex, right)
        if midindex == split_index:
            break
        elif midindex < split_index:
            left = midindex + 1
        else:
            right = midindex - 1

    return 0


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


#  TODO - make `dist` and `rdist` inline methods
#       - count distance calls & other stats
#       - make class pickleable
#       - breadth-first query using a priority queue of nodes
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
    # XXX: double-check that these are used correctly
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

    cdef inline DTYPE_t dist(self, DTYPE_t[:, ::1] X1, ITYPE_t i1,
                             DTYPE_t[:, ::1] X2, ITYPE_t i2):
        if self.euclidean:
            return euclidean_dist(X1, i1, X2, i2)
        else:
            return self.dm.dist(X1, i1, X2, i2)

    cdef inline DTYPE_t rdist(self, DTYPE_t[:, ::1] X1, ITYPE_t i1,
                              DTYPE_t[:, ::1] X2, ITYPE_t i2):
        if self.euclidean:
            return euclidean_rdist(X1, i1, X2, i2)
        else:
            return self.dm.rdist(X1, i1, X2, i2)

    cdef void _recursive_build(self, ITYPE_t i_node,
                               ITYPE_t idx_start, ITYPE_t idx_end):
        cdef ITYPE_t imax
        cdef ITYPE_t n_features = self.data.shape[1]
        cdef ITYPE_t n_points = idx_end - idx_start
        cdef ITYPE_t n_mid = idx_start + n_points / 2

        # initialize node data
        self.init_node(i_node, idx_start, idx_end)

        if 2 * i_node + 1 >= self.n_nodes:
            self.node_data[i_node].is_leaf = True
            if idx_end - idx_start > 2 * self.leaf_size:
                # this shouldn't happen if our memory allocation is correct
                # we'll proactively prevent memory errors, but raise a
                # warning saying we're doing so.
                warnings.warn("Internal: memory layout is flawed: "
                              "not enough nodes allocated")

        elif idx_end - idx_start < 2:
            # again, this shouldn't happen if our memory allocation
            # is correct.  Raise a warning.
            warnings.warn("Internal: memory layout is flawed: "
                          "too many nodes allocated")
            self.node_data[i_node].is_leaf = True

        else: 
            # split node and recursively construct child nodes.
            self.node_data[i_node].is_leaf = False
            i_max = find_split_dim(self.data, self.idx_array,
                                   idx_start, idx_end)
            partition_indices(self.data, self.idx_array,
                              i_max, idx_start, n_mid, idx_end)

            self._recursive_build(2 * i_node + 1, idx_start, n_mid)
            self._recursive_build(2 * i_node + 2, n_mid, idx_end)

    def query(self, X, k=1, return_distance=True, dualtree=False):
        """
        query(X, k=1, return_distance=True)

        query the Tree for the k nearest neighbors

        Parameters
        ----------
        X : array-like, last dimension self.n_features
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
        # >>> dist, ind = ball_tree.query(X[0], k=3)
        # >>> print ind  # indices of 3 closest neighbors
        # [0 3 1]
        # >>> print dist  # distances to 3 closest neighbors
        # [ 0.          0.19662693  0.29473397]
        """
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

        # initialize heap for neighbors
        cdef NeighborsHeap heap = NeighborsHeap(Xarr.shape[0], k)

        # bounds is needed for the dual tree algorithm
        cdef DTYPE_t[::1] bounds

        self.n_trims = 0
        self.n_leaves = 0
        self.n_splits = 0

        if dualtree:
            # build a tree on query data with the same metric as self
            # XXX: make sure this is correct, and allow passing a tree
            other = self.__class__(Xarr, leaf_size=self.leaf_size)
            reduced_dist_LB = self.min_rdist_dual(0, other, 0)

            bounds = np.inf + np.zeros(other.node_data.shape[0])

            self._query_dual(0, other, 0, bounds, heap, reduced_dist_LB)

        else:
            for i in range(Xarr.shape[0]):
                reduced_dist_LB = self.min_rdist(0, Xarr, i)
                self._query_one(0, i, Xarr, heap, reduced_dist_LB)

        distances, indices = heap.get_arrays(sort=True)
        distances = self.dm.rdist_to_dist_arr(distances)

        # deflatten results
        if return_distance:
            return (distances.reshape((X.shape[:-1]) + (k,)),
                    indices.reshape((X.shape[:-1]) + (k,)))
        else:
            return indices.reshape((X.shape[:-1]) + (k,))

    cdef void _query_one(self, ITYPE_t i_node, ITYPE_t i_pt,
                         DTYPE_t[:, ::1] points, NeighborsHeap heap,
                         DTYPE_t reduced_dist_LB):
        cdef NodeData_t node_info = self.node_data[i_node]

        cdef DTYPE_t dist_pt, reduced_dist_LB_1, reduced_dist_LB_2
        cdef ITYPE_t i, i1, i2

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
                dist_pt = self.rdist(points, i_pt,
                                     self.data, self.idx_array[i])

                if dist_pt < heap.largest(i_pt):
                    heap.push(i_pt, dist_pt, self.idx_array[i])

        #------------------------------------------------------------
        # Case 3: Node is not a leaf.  Recursively query subnodes
        #         starting with the closest
        else:
            self.n_splits += 1
            i1 = 2 * i_node + 1
            i2 = i1 + 1
            reduced_dist_LB_1 = self.min_rdist(i1, points, i_pt)
            reduced_dist_LB_2 = self.min_rdist(i2, points, i_pt)

            # recursively query subnodes
            if reduced_dist_LB_1 <= reduced_dist_LB_2:
                self._query_one(i1, i_pt, points, heap, reduced_dist_LB_1)
                self._query_one(i2, i_pt, points, heap, reduced_dist_LB_2)
            else:
                self._query_one(i2, i_pt, points, heap, reduced_dist_LB_2)
                self._query_one(i1, i_pt, points, heap, reduced_dist_LB_1)

    cdef void _query_dual(BallTree self, ITYPE_t i_node1,
                          BallTree other, ITYPE_t i_node2,
                          DTYPE_t[::1] bounds, NeighborsHeap heap,
                          DTYPE_t reduced_dist_LB):
        cdef ITYPE_t n_features = self.data.shape[1]
        cdef NodeData_t node_info1 = self.node_data[i_node1]
        cdef NodeData_t node_info2 = other.node_data[i_node2]

        cdef DTYPE_t dist_pt, reduced_dist_LB_1, reduced_dist_LB_2
        cdef ITYPE_t i, i1, i2, i_pt

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
                    dist_pt = self.rdist(self.data, self.idx_array[i1],
                                         other.data, i_pt)
                    if dist_pt < heap.largest(i_pt):
                        heap.push(i_pt, dist_pt, self.idx_array[i1])
                
                # keep track of node bound
                bounds[i_node2] = fmax(bounds[i_node2],
                                       heap.largest(i_pt))
            
        #------------------------------------------------------------
        # Case 3a: only node 1 is a leaf: split node 2 and recursively
        #          query, starting with the nearest node
        elif node_info1.is_leaf:
            reduced_dist_LB_1 = self.min_rdist_dual(i_node1,
                                                    other, 2 * i_node2 + 1)
            reduced_dist_LB_2 = self.min_rdist_dual(i_node1,
                                                    other, 2 * i_node2 + 2)

            if reduced_dist_LB_1 < reduced_dist_LB_2:
                self._query_dual(i_node1, other, 2 * i_node2 + 1,
                                 bounds, heap, reduced_dist_LB_1)
                self._query_dual(i_node1, other, 2 * i_node2 + 2,
                                 bounds, heap, reduced_dist_LB_2)
            else:
                self._query_dual(i_node1, other, 2 * i_node2 + 2,
                                 bounds, heap, reduced_dist_LB_2)
                self._query_dual(i_node1, other, 2 * i_node2 + 1,
                                 bounds, heap, reduced_dist_LB_1)
            
            # update node bound information
            bounds[i_node2] = fmax(bounds[2 * i_node2 + 1],
                                   bounds[2 * i_node2 + 2])
            
        #------------------------------------------------------------
        # Case 3b: only node 2 is a leaf: split node 1 and recursively
        #          query, starting with the nearest node
        elif node_info2.is_leaf:
            reduced_dist_LB_1 = self.min_rdist_dual(2 * i_node1 + 1,
                                                    other, i_node2)
            reduced_dist_LB_2 = self.min_rdist_dual(2 * i_node1 + 2,
                                                    other, i_node2)

            if reduced_dist_LB_1 < reduced_dist_LB_2:
                self._query_dual(2 * i_node1 + 1, other, i_node2,
                                 bounds, heap, reduced_dist_LB_1)
                self._query_dual(2 * i_node1 + 2, other, i_node2,
                                 bounds, heap, reduced_dist_LB_2)
            else:
                self._query_dual(2 * i_node1 + 2, other, i_node2,
                                 bounds, heap, reduced_dist_LB_2)
                self._query_dual(2 * i_node1 + 1, other, i_node2,
                                 bounds, heap, reduced_dist_LB_1)
        
        #------------------------------------------------------------
        # Case 4: neither node is a leaf:
        #         split both and recursively query all four pairs
        else:
            reduced_dist_LB_1 = self.min_rdist_dual(2 * i_node1 + 1,
                                                    other, 2 * i_node2 + 1)
            reduced_dist_LB_2 = self.min_rdist_dual(2 * i_node1 + 2,
                                                    other, 2 * i_node2 + 1)

            if reduced_dist_LB_1 < reduced_dist_LB_2:
                self._query_dual(2 * i_node1 + 1, other, 2 * i_node2 + 1,
                                 bounds, heap, reduced_dist_LB_1)
                self._query_dual(2 * i_node1 + 2, other, 2 * i_node2 + 1,
                                 bounds, heap, reduced_dist_LB_2)
            else:
                self._query_dual(2 * i_node1 + 2, other, 2 * i_node2 + 1,
                                 bounds, heap, reduced_dist_LB_2)
                self._query_dual(2 * i_node1 + 1, other, 2 * i_node2 + 1,
                                 bounds, heap, reduced_dist_LB_1)

            reduced_dist_LB_1 = self.min_rdist_dual(2 * i_node1 + 1,
                                                    other, 2 * i_node2 + 2)
            reduced_dist_LB_2 = self.min_rdist_dual(2 * i_node1 + 2,
                                                    other, 2 * i_node2 + 2)
            if reduced_dist_LB_1 < reduced_dist_LB_2:
                self._query_dual(2 * i_node1 + 1, other, 2 * i_node2 + 2,
                                 bounds, heap, reduced_dist_LB_1)
                self._query_dual(2 * i_node1 + 2, other, 2 * i_node2 + 2,
                                 bounds, heap, reduced_dist_LB_2)
            else:
                self._query_dual(2 * i_node1 + 2, other, 2 * i_node2 + 2,
                                 bounds, heap, reduced_dist_LB_2)
                self._query_dual(2 * i_node1 + 1, other, 2 * i_node2 + 2,
                                 bounds, heap, reduced_dist_LB_1)
            
            # update node bound information
            bounds[i_node2] = fmax(bounds[2 * i_node2 + 1],
                                   bounds[2 * i_node2 + 2])

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

        cdef DTYPE_t[::1] centroid = self.centroids[i_node]

        # determine Node centroid -- could we tie into a BLAS function?
        for j in range(n_features):
            centroid[j] = 0

        for i in range(idx_start, idx_end):
            for j in range(n_features):
                centroid[j] += self.data[self.idx_array[i], j]

        for j in range(n_features):
            centroid[j] /= n_points

        # determine Node radius
        radius = 0
        for i in range(idx_start, idx_end):
            radius = fmax(radius,
                          self.rdist(self.centroids, i_node, 
                                     self.data, self.idx_array[i]))

        self.node_data[i_node].radius = self.dm.rdist_to_dist(radius)
        self.node_data[i_node].idx_start = idx_start
        self.node_data[i_node].idx_end = idx_end

    cdef inline DTYPE_t min_dist(BallTree self, ITYPE_t i_node,
                                 DTYPE_t[:, ::1] p, ITYPE_t i_p):
        cdef DTYPE_t dist_pt
        dist_pt = self.dist(p, i_p, self.centroids, i_node)
        return fmax(0, dist_pt - self.node_data[i_node].radius)
            
    cdef inline DTYPE_t max_dist(BallTree self, ITYPE_t i_node,
                                 DTYPE_t[:, ::1] p, ITYPE_t i_p):
        cdef DTYPE_t dist_pt
        dist_pt = self.dist(p, i_p, self.centroids, i_node)
        return dist_pt + self.node_data[i_node].radius

    cdef inline DTYPE_t min_rdist(BallTree self, ITYPE_t i_node,
                                  DTYPE_t[:, ::1] p, ITYPE_t i_p):
        return self.dm.dist_to_rdist(self.min_dist(i_node, p, i_p))
            
    cdef inline DTYPE_t max_rdist(BallTree self, ITYPE_t i_node,
                                  DTYPE_t[:, ::1] p, ITYPE_t i_p):
        return self.dm.dist_to_rdist(self.max_dist(i_node, p, i_p))

    cdef inline DTYPE_t min_dist_dual(BallTree self, ITYPE_t i_node1,
                                      BallTree other, ITYPE_t i_node2):
        cdef DTYPE_t dist_pt
        dist_pt = self.dist(self.centroids, i_node1,
                            other.centroids, i_node2)
        return fmax(0, (dist_pt
                        - self.node_data[i_node1].radius
                        - other.node_data[i_node2].radius))

    cdef inline DTYPE_t max_dist_dual(BallTree self, ITYPE_t i_node1,
                                      BallTree other, ITYPE_t i_node2):
        cdef DTYPE_t dist_pt
        dist_pt = self.dist(self.centroids, i_node1,
                            other.centroids, i_node2)
        return (dist_pt
                + self.node_data[i_node1].radius
                + other.node_data[i_node2].radius)
    
    cdef inline DTYPE_t min_rdist_dual(BallTree self, ITYPE_t i_node1,
                                       BallTree other, ITYPE_t i_node2):
        return self.dm.dist_to_rdist(self.min_dist_dual(i_node1,
                                                        other, i_node2))

    cdef inline DTYPE_t max_rdist_dual(BallTree self, ITYPE_t i_node1,
                                       BallTree other, ITYPE_t i_node2):
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
            D[i1, i2] = eucl_dist.dist(X, i1, Y, i2)
    return np.asarray(D)


def euclidean_pairwise_polymorphic(DTYPE_t[:, ::1] X, DTYPE_t[:, ::1] Y):
    cdef DistanceMetric eucl_dist = EuclideanDistance()

    assert X.shape[1] == Y.shape[1]
    cdef DTYPE_t[:, ::1] D = np.zeros((X.shape[0], Y.shape[0]), dtype=DTYPE)

    for i1 in range(X.shape[0]):
        for i2 in range(Y.shape[0]):
            D[i1, i2] = eucl_dist.dist(X, i1, Y, i2)
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
    """In-place simultaneous sort the given row of the arrays"""
    assert distances.shape[0] == indices.shape[0]
    assert distances.shape[1] == indices.shape[1]

    cdef ITYPE_t row
    for row in range(distances.shape[0]):
        _simultaneous_sort(distances, indices, row, 0, distances.shape[1])

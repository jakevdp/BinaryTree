#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
"""
Binary Tree
-----------
This is the Abstract Base Class for the Ball Tree and KD Tree
"""
import warnings
import numpy as np
from sklearn.utils import array2d

from libc.math cimport fmax, fmin, fabs, sqrt
cimport numpy as np
cimport cython

from distmetrics cimport DistanceMetric
from distmetrics import Distance

from tree_utils cimport MaxHeap, partition_indices, find_split_dim

#####################################################################
# global types and variables
from typedefs cimport DTYPE_t, ITYPE_t
from typedefs import DTYPE, ITYPE
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
# Fase inline distance function for Euclidean case
cdef inline DTYPE_t dist(DTYPE_t[:, ::1] X1, ITYPE_t i1,
                         DTYPE_t[:, ::1] X2, ITYPE_t i2):
    cdef ITYPE_t n_features = X1.shape[1]
    cdef DTYPE_t tmp, d=0
    for j in range(n_features):
        tmp = X1[i1, j] - X2[i2, j]
        d += tmp * tmp
    return sqrt(d)

cdef inline DTYPE_t rdist(DTYPE_t[:, ::1] X1, ITYPE_t i1,
                          DTYPE_t[:, ::1] X2, ITYPE_t i2):
    cdef ITYPE_t n_features = X1.shape[1]
    cdef DTYPE_t tmp, d=0
    for j in range(n_features):
        tmp = X1[i1, j] - X2[i2, j]
        d += tmp * tmp
    return d


######################################################################
# BinaryTree Abstract Base Class
cdef class BinaryTree:
    """Abstract base class for binary tree objects"""
    cdef readonly DTYPE_t[:, ::1] data
    cdef public ITYPE_t[::1] idx_array
    cdef public NodeData_t[::1] node_data

    # used for BallTree
    cdef public DTYPE_t[:, ::1] centroids_arr

    # used for KDTree
    cdef public DTYPE_t[:, ::1] lower_bounds
    cdef public DTYPE_t[:, ::1] upper_bounds

    cdef ITYPE_t leaf_size
    cdef ITYPE_t n_levels
    cdef ITYPE_t n_nodes

    cdef MaxHeap heap
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
        self.centroids_arr = np.empty((0, 1), dtype=DTYPE)
        self.lower_bounds = np.empty((0, 1), dtype=DTYPE, order='C')
        self.upper_bounds = np.empty((0, 1), dtype=DTYPE, order='C')

        self.leaf_size = 0
        self.n_levels = 0
        self.n_nodes = 0
        self.n_calls = 0
        self.euclidean = False

    def __init__(self, DTYPE_t[:, ::1] data,
                 leaf_size=20, metric='minkowski', **kwargs):
        if self.__class__ is BinaryTree:
            raise NotImplementedError("BinaryTree is an abstract class")
        self.data = data
        self.idx_array = np.arange(data.shape[0], dtype=ITYPE)
        self.node_data = np.zeros(data.shape[0], dtype=NodeData)
        self.leaf_size = leaf_size
        self.dm = Distance(metric, **kwargs)
        self.euclidean = (self.dm.__class__.__name__ == 'EuclideanDistance')

        self.heap = MaxHeap()

        # validate data
        if self.data.size == 0:
            raise ValueError("X is an empty array")

        if leaf_size < 1:
            raise ValueError("leaf_size must be greater than or equal to 1")
        
        cdef ITYPE_t n_samples = self.data.shape[0]
        cdef ITYPE_t n_features = self.data.shape[1]
        self.compute_node_count()

        # allocate arrays for storage
        self.idx_array = np.arange(n_samples, dtype=ITYPE)
        self.node_data = np.zeros(self.n_nodes, dtype=NodeData)

        # Allocate tree-specific data from TreeBase
        self.allocate_data(self.n_nodes, n_features)        
        self._recursive_build(0, 0, n_samples)

    cdef void compute_node_count(self):
        # determine number of levels in the tree, and from this
        # the number of nodes in the tree.  This results in leaf nodes
        # with numbers of points betweeen leaf_size and 2 * leaf_size
        # (see module-level doc string for details)
        cdef ITYPE_t n_samples = self.data.shape[0]
        self.n_levels = np.log2(fmax(1, (n_samples - 1) / self.leaf_size)) + 1
        self.n_nodes = (2 ** self.n_levels) - 1

    def get_arrays(self):
        return map(np.asarray, (self.data, self.idx_array, self.node_data))

    def get_tree_stats(self):
        return (self.n_trims, self.n_leaves, self.n_splits)

    def reset_n_calls(self):
        self.n_calls = 0

    def get_n_calls(self):
        return self.n_calls

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
                # we'll proactively prevent memory errors, but raise a warning
                # saying we're doing so.
                warnings.warn("Internal: memory layout is flawed: "
                              "not enough nodes allocated")

        elif idx_end - idx_start < 2:
            # again, this shouldn't happen if our memory allocation is correct
            warnings.warn("Internal: memory layout is flawed: "
                          "too many nodes allocated")
            self.node_data[i_node].is_leaf = True

        else:  # split node and recursively construct child nodes.
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
        X = array2d(X, dtype=DTYPE, order='C')

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

        # allocate distances and indices for return
        distances = np.empty((Xarr.shape[0], k),
                             dtype=DTYPE)
        distances.fill(np.inf)
        indices = np.zeros((Xarr.shape[0], k),
                           dtype=ITYPE)

        cdef DTYPE_t[:, ::1] distances_arr = distances
        cdef ITYPE_t[:, ::1] indices_arr = indices
        cdef DTYPE_t[::1] bounds

        self.n_trims = 0
        self.n_leaves = 0
        self.n_splits = 0

        if dualtree:
            # build a tree on query data with the same metric as self
            # XXX: make sure this is correct, and allow passing a tree
            other = self.__class__(X, leaf_size=self.leaf_size)
            print other
            reduced_dist_LB = self.min_rdist_dual(0, other, 0)

            bounds = np.inf + np.zeros(other.node_data.shape[0])

            self._query_dual(0, other, 0,
                             distances_arr, indices_arr, bounds,
                             reduced_dist_LB)
            for i in range(other.data.shape[0]):
                self.heap.wrap(distances_arr[i], indices_arr[i])
                self.heap.sort()

        else:
            for i in range(Xarr.shape[0]):
                reduced_dist_LB = self.min_rdist(0, Xarr, i)
                self._query_one(0, i, Xarr, distances_arr,
                                indices_arr, reduced_dist_LB)
                self.heap.sort()

        distances = self.dm.rdist_to_dist(distances)

        # deflatten results
        if return_distance:
            return (distances.reshape((X.shape[:-1]) + (k,)),
                    indices.reshape((X.shape[:-1]) + (k,)))
        else:
            return indices.reshape((X.shape[:-1]) + (k,))


    @cython.boundscheck(False)
    cdef void _query_one(self, ITYPE_t i_node, ITYPE_t i_pt,
                         DTYPE_t[:, ::1] points,
                         DTYPE_t[:, ::1] distances,
                         ITYPE_t[:, ::1] indices,
                         DTYPE_t reduced_dist_LB):
        cdef DTYPE_t[::1] pt = points[i_pt]
        cdef DTYPE_t[::1] dist = distances[i_pt]
        cdef ITYPE_t[::1] ind = indices[i_pt]
        cdef NodeData_t node_info = self.node_data[i_node]

        cdef DTYPE_t dist_pt, reduced_dist_LB_1, reduced_dist_LB_2
        cdef ITYPE_t i, i1, i2

        # Initialize the heap
        self.heap.wrap(dist, ind)

        #------------------------------------------------------------
        # Case 1: query point is outside node radius:
        #         trim it from the query
        if reduced_dist_LB > self.heap.largest():
            self.n_trims += 1

        #------------------------------------------------------------
        # Case 2: this is a leaf node.  Update set of nearby points
        elif node_info.is_leaf:
            self.n_leaves += 1
            if self.euclidean:
                for i in range(node_info.idx_start, node_info.idx_end):
                    dist_pt = rdist(points, i_pt,
                                    self.data, self.idx_array[i])

                    if dist_pt < self.heap.largest():
                        self.heap.push(dist_pt, self.idx_array[i])
            else:
                for i in range(node_info.idx_start, node_info.idx_end):
                    dist_pt = self.dm.rdist(points, i_pt,
                                            self.data, self.idx_array[i])

                    if dist_pt < self.heap.largest():
                        self.heap.push(dist_pt, self.idx_array[i])

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
                self._query_one(i1, i_pt, points, distances, indices,
                                reduced_dist_LB_1)
                self._query_one(i2, i_pt, points, distances, indices,
                                reduced_dist_LB_2)
            else:
                self._query_one(i2, i_pt, points, distances, indices,
                                reduced_dist_LB_2)
                self._query_one(i1, i_pt, points, distances, indices,
                                reduced_dist_LB_1)

    @cython.boundscheck(False)
    cdef void _query_dual(BinaryTree self, ITYPE_t i_node1,
                          BinaryTree other, ITYPE_t i_node2,
                          DTYPE_t[:, ::1] distances,
                          ITYPE_t[:, ::1] indices,
                          DTYPE_t[::1] bounds,
                          DTYPE_t reduced_dist_LB):
        cdef ITYPE_t n_features = self.data.shape[1]
        cdef NodeData_t node_info1 = self.node_data[i_node1]
        cdef NodeData_t node_info2 = other.node_data[i_node2]

        cdef DTYPE_t dist_pt, reduced_dist_LB_1, reduced_dist_LB_2
        cdef ITYPE_t i, i1, i2

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
                self.heap.wrap(distances[other.idx_array[i2]],
                               indices[other.idx_array[i2]])

                if self.heap.largest() <= reduced_dist_LB:
                    continue

                for i1 in range(node_info1.idx_start, node_info1.idx_end):
                    dist_pt = self.dm.rdist(self.data, self.idx_array[i1],
                                            other.data, other.idx_array[i2])
                    if dist_pt < self.heap.largest():
                        self.heap.push(dist_pt, self.idx_array[i1])
                
                # keep track of node bound
                bounds[i_node2] = fmax(bounds[i_node2], self.heap.largest())
            
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
                                 distances, indices, bounds,
                                 reduced_dist_LB_1)
                self._query_dual(i_node1, other, 2 * i_node2 + 2,
                                 distances, indices, bounds,
                                 reduced_dist_LB_2)
            else:
                self._query_dual(i_node1, other, 2 * i_node2 + 2,
                                 distances, indices, bounds,
                                 reduced_dist_LB_2)
                self._query_dual(i_node1, other, 2 * i_node2 + 1,
                                 distances, indices, bounds,
                                 reduced_dist_LB_1)
            
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
                                 distances, indices, bounds,
                                 reduced_dist_LB_1)
                self._query_dual(2 * i_node1 + 2, other, i_node2,
                                 distances, indices, bounds,
                                 reduced_dist_LB_2)
            else:
                self._query_dual(2 * i_node1 + 2, other, i_node2,
                                 distances, indices, bounds,
                                 reduced_dist_LB_2)
                self._query_dual(2 * i_node1 + 1, other, i_node2,
                                 distances, indices, bounds,
                                 reduced_dist_LB_1)
        
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
                                 distances, indices, bounds,
                                 reduced_dist_LB_1)
                self._query_dual(2 * i_node1 + 2, other, 2 * i_node2 + 1,
                                 distances, indices, bounds,
                                 reduced_dist_LB_2)
            else:
                self._query_dual(2 * i_node1 + 2, other, 2 * i_node2 + 1,
                                 distances, indices, bounds,
                                 reduced_dist_LB_2)
                self._query_dual(2 * i_node1 + 1, other, 2 * i_node2 + 1,
                                 distances, indices, bounds,
                                 reduced_dist_LB_1)

            reduced_dist_LB_1 = self.min_rdist_dual(2 * i_node1 + 1,
                                                    other, 2 * i_node2 + 2)
            reduced_dist_LB_2 = self.min_rdist_dual(2 * i_node1 + 2,
                                                    other, 2 * i_node2 + 2)
            if reduced_dist_LB_1 < reduced_dist_LB_2:
                self._query_dual(2 * i_node1 + 1, other, 2 * i_node2 + 2,
                                 distances, indices, bounds,
                                 reduced_dist_LB_1)
                self._query_dual(2 * i_node1 + 2, other, 2 * i_node2 + 2,
                                 distances, indices, bounds,
                                 reduced_dist_LB_2)
            else:
                self._query_dual(2 * i_node1 + 2, other, 2 * i_node2 + 2,
                                 distances, indices, bounds,
                                 reduced_dist_LB_2)
                self._query_dual(2 * i_node1 + 1, other, 2 * i_node2 + 2,
                                 distances, indices, bounds,
                                 reduced_dist_LB_1)
            
            # update node bound information
            bounds[i_node2] = fmax(bounds[2 * i_node2 + 1],
                                   bounds[2 * i_node2 + 2])
                          
        

    #----------------------------------------------------------------------
    # These should be specialized in derived classes
    cdef void allocate_data(BinaryTree self, ITYPE_t n_nodes,
                            ITYPE_t n_features):
        raise NotImplementedError()

    cdef void init_node(BinaryTree self, ITYPE_t i_node,
                        ITYPE_t idx_start, ITYPE_t idx_end):
        raise NotImplementedError()

    cdef DTYPE_t min_dist(BinaryTree self, ITYPE_t i_node,
                          DTYPE_t[:, ::1] p, ITYPE_t i_p):
        raise NotImplementedError()

    cdef DTYPE_t min_rdist(BinaryTree self, ITYPE_t i_node,
                           DTYPE_t[:, ::1] p, ITYPE_t i_p):
        raise NotImplementedError()

    cdef DTYPE_t max_dist(BinaryTree self, ITYPE_t i_node,
                          DTYPE_t[:, ::1] p, ITYPE_t i_p):
        raise NotImplementedError()

    cdef DTYPE_t max_rdist(BinaryTree self, ITYPE_t i_node,
                           DTYPE_t[:, ::1] p, ITYPE_t i_p):
        raise NotImplementedError()

    cdef DTYPE_t min_dist_dual(BinaryTree self, ITYPE_t i_node1,
                               BinaryTree other, ITYPE_t i_node2):
        raise NotImplementedError()

    cdef DTYPE_t min_rdist_dual(BinaryTree self, ITYPE_t i_node1,
                                BinaryTree other, ITYPE_t i_node2):
        raise NotImplementedError()

    cdef DTYPE_t max_dist_dual(BinaryTree self, ITYPE_t i_node1,
                               BinaryTree other, ITYPE_t i_node2):
        raise NotImplementedError()

    cdef DTYPE_t max_rdist_dual(BinaryTree self, ITYPE_t i_node1,
                                BinaryTree other, ITYPE_t i_node2):
        raise NotImplementedError()


@cython.final
cdef class BallTree(BinaryTree):
    """Ball Tree for nearest neighbor queries"""
    def get_arrays(self):
        return map(np.asarray, (self.data, self.idx_array, self.node_data,
                                self.centroids_arr))

    cdef void allocate_data(self, ITYPE_t n_nodes, ITYPE_t n_features):
        self.centroids_arr = np.zeros((n_nodes, n_features), dtype=DTYPE)

    cdef void init_node(self, ITYPE_t i_node,
                        ITYPE_t idx_start, ITYPE_t idx_end):
        cdef ITYPE_t n_features = self.data.shape[1]
        cdef ITYPE_t n_points = idx_end - idx_start

        cdef ITYPE_t i, j
        cdef DTYPE_t radius
        cdef DTYPE_t *this_pt

        cdef DTYPE_t[::1] centroid = self.centroids_arr[i_node]

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
        if self.euclidean:
            for i in range(idx_start, idx_end):
                radius = fmax(radius,
                              dist(self.centroids_arr, i_node, 
                                   self.data, self.idx_array[i]))
        else:
            for i in range(idx_start, idx_end):
                radius = fmax(radius,
                              self.dm.dist(self.centroids_arr, i_node, 
                                           self.data, self.idx_array[i]))

        self.node_data[i_node].radius = radius
        self.node_data[i_node].idx_start = idx_start
        self.node_data[i_node].idx_end = idx_end

    cdef DTYPE_t min_dist(BallTree self, ITYPE_t i_node,
                          DTYPE_t[:, ::1] p, ITYPE_t i_p):
        cdef DTYPE_t dist_pt
        if self.euclidean:
            dist_pt = dist(p, i_p, self.centroids_arr, i_node)
        else:
            dist_pt = self.dm.dist(p, i_p, self.centroids_arr, i_node)
        return fmax(0, dist_pt - self.node_data[i_node].radius)
            
    cdef DTYPE_t max_dist(BallTree self, ITYPE_t i_node,
                           DTYPE_t[:, ::1] p, ITYPE_t i_p):
        cdef DTYPE_t dist_pt
        if self.euclidean:
            dist_pt = dist(p, i_p, self.centroids_arr, i_node)
        else:
            dist_pt = self.dm.dist(p, i_p, self.centroids_arr, i_node)
        return dist_pt + self.node_data[i_node].radius

    cdef DTYPE_t min_rdist(BallTree self, ITYPE_t i_node,
                           DTYPE_t[:, ::1] p, ITYPE_t i_p):
        cdef DTYPE_t tmp = self.min_dist(i_node, p, i_p)
        return tmp * tmp
            
    cdef DTYPE_t max_rdist(BallTree self, ITYPE_t i_node,
                           DTYPE_t[:, ::1] p, ITYPE_t i_p):
        cdef DTYPE_t tmp = self.max_dist(i_node, p, i_p)
        return tmp * tmp

    cdef DTYPE_t min_dist_dual(BallTree self, ITYPE_t i_node1,
                               BinaryTree other, ITYPE_t i_node2):
        cdef DTYPE_t dist_pt
        if self.euclidean:
            dist_pt = dist(self.centroids_arr, i_node1,
                           other.centroids_arr, i_node2)
        else:
            dist_pt = self.dm.dist(self.centroids_arr, i_node1,
                                   other.centroids_arr, i_node2)
        return fmax(0, (dist_pt
                        - self.node_data[i_node1].radius
                        - other.node_data[i_node2].radius))
    
    cdef DTYPE_t min_rdist_dual(BallTree self, ITYPE_t i_node1,
                                BinaryTree other, ITYPE_t i_node2):
        cdef DTYPE_t tmp = self.min_dist_dual(i_node1, other, i_node2)
        return tmp * tmp

    cdef DTYPE_t max_dist_dual(BallTree self, ITYPE_t i_node1,
                               BinaryTree other, ITYPE_t i_node2):
        cdef DTYPE_t dist_pt
        if self.euclidean:
            dist_pt = dist(self.centroids_arr, i_node1,
                           other.centroids_arr, i_node2)
        else:
            dist_pt = self.dm.dist(self.centroids_arr, i_node1,
                                   other.centroids_arr, i_node2)
        return (dist_pt
                + self.node_data[i_node1].radius
                + other.node_data[i_node2].radius)

    cdef DTYPE_t max_rdist_dual(BallTree self, ITYPE_t i_node1,
                                BinaryTree other, ITYPE_t i_node2):
        cdef DTYPE_t tmp = self.max_dist_dual(i_node1, other, i_node2)
        return tmp * tmp


cdef class KDTree(BinaryTree):
    """KD Tree for nearest neighbor queries"""
    def __init__(self, DTYPE_t[:, ::1] data,
                 leaf_size=20, metric='minkowski', **kwargs):
        self.dm = Distance(metric, **kwargs)
        if not self.dm.axis_aligned:
            raise ValueError('KDTree can only be used for axis-aligned '
                             'distance metrics.  Use BallTree instead.')
        BinaryTree.__init__(self, data, leaf_size, metric, **kwargs)
        

    def get_arrays(self):
        return map(np.asarray, (self.data, self.idx_array, self.node_data,
                                self.lower_bounds, self.upper_bounds))

    cdef void allocate_data(self, ITYPE_t n_nodes, ITYPE_t n_features):
        self.lower_bounds = np.zeros((n_nodes, n_features),
                                         dtype=DTYPE, order='C')
        self.upper_bounds = np.zeros((n_nodes, n_features),
                                         dtype=DTYPE, order='C')

    cdef void init_node(self, ITYPE_t i_node,
                        ITYPE_t idx_start, ITYPE_t idx_end):
        cdef ITYPE_t n_features = self.data.shape[1]
        cdef ITYPE_t i, j, idx_i

        # determine Node bounds
        for j in range(n_features):
            self.lower_bounds[i_node, j] = INF
            self.upper_bounds[i_node, j] = -INF

        for i in range(idx_start, idx_end):
            idx_i = self.idx_array[i]
            for j in range(n_features):
                self.lower_bounds[i_node, j] =\
                    fmin(self.lower_bounds[i_node, j],
                         self.data[idx_i, j])
                self.upper_bounds[i_node, j] =\
                    fmax(self.upper_bounds[i_node, j],
                         self.data[idx_i, j])

        self.node_data[i_node].idx_start = idx_start
        self.node_data[i_node].idx_end = idx_end

    cdef DTYPE_t min_rdist(self, ITYPE_t i_node,
                          DTYPE_t[:, ::1] p, ITYPE_t i_p):
        cdef ITYPE_t n_features = self.data.shape[1]
        cdef DTYPE_t d, d_lo, d_hi, rdist=0.0
        cdef ITYPE_t j

        # here we'll use the fact that x + abs(x) = 2 * max(x, 0)
        for j in range(n_features):
            d_lo = self.lower_bounds[i_node, j] - p[i_p, j]
            d_hi = p[i_p, j] - self.upper_bounds[i_node, j]
            d = (d_lo + fabs(d_lo)) + (d_hi + fabs(d_hi))
            rdist += pow(d, self.dm.p)

        return rdist / pow(2, self.dm.p)

    cdef DTYPE_t min_dist(self, ITYPE_t i_node,
                           DTYPE_t[:, ::1] p, ITYPE_t i_p):
        return pow(self.min_rdist(i_node, p, i_p), 1. / self.dm.p)

    cdef DTYPE_t max_rdist(self, ITYPE_t i_node,
                          DTYPE_t[:, ::1] p, ITYPE_t i_p):
        cdef ITYPE_t n_features = self.data.shape[1]

        cdef DTYPE_t d, d_lo, d_hi, rdist=0.0
        cdef ITYPE_t j

        for j in range(n_features):
            d_lo = self.lower_bounds[i_node, j] - p[i_p, j]
            d_hi = p[i_p, j] - self.upper_bounds[i_node, j]
            rdist += pow(fmax(d_lo, d_hi), self.dm.p)

        return rdist

    cdef DTYPE_t max_dist(self, ITYPE_t i_node,
                           DTYPE_t[:, ::1] p, ITYPE_t i_p):
        return pow(self.max_rdist(i_node, p, i_p), 1. / self.dm.p)

#!python
"""Binary Tree

This is the Abstract Base Class for the Ball Tree and KD Tree
"""
import numpy as np

from libc.math cimport fmax, fmin, fabs

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
# BinaryTree Abstract Base Class
cdef class _BinaryTree:
    """Abstract base class for binary tree objects"""
    cdef readonly DTYPE_t[:, ::1] data
    cdef ITYPE_t[::1] idx_array
    cdef NodeData_t[::1] node_data_arr

    cdef ITYPE_t leaf_size
    cdef ITYPE_t n_levels
    cdef ITYPE_t n_nodes

    cdef MaxHeap heap
    cdef DistanceMetric dm

    # variables to keep track of building & querying stats
    cdef int n_trims
    cdef int n_leaves
    cdef int n_splits

    # ball tree data
    cdef DTYPE_t[:, ::1] centroids_arr

    def __init__(self, DTYPE_t[:, ::1] data,
                 leaf_size=20, metric='minkowski', p=2, **kwargs):
        self.data = data
        self.idx_array = np.arange(data.shape[0], dtype=ITYPE)
        self.node_data_arr = np.zeros(data.shape[0], dtype=NodeData)
        self.leaf_size = leaf_size
        self.dm = Distance(metric, **kwargs)
        self.heap = MaxHeap()

        # validate data
        if self.data.size == 0:
            raise ValueError("X is an empty array")

        if leaf_size < 1:
            raise ValueError("leaf_size must be greater than or equal to 1")
        
        cdef ITYPE_t n_samples = self.data.shape[0]
        cdef ITYPE_t n_features = self.data.shape[1]

        # determine number of levels in the tree, and from this
        # the number of nodes in the tree.  This results in leaf nodes
        # with numbers of points betweeen leaf_size and 2 * leaf_size
        # (see module-level doc string for details)
        self.n_levels = np.log2(fmax(1, (n_samples - 1) / self.leaf_size)) + 1
        self.n_nodes = (2 ** self.n_levels) - 1

        # allocate arrays for storage
        self.idx_array = np.arange(n_samples, dtype=ITYPE)
        self.node_data_arr = np.zeros(self.n_nodes)

        # Allocate tree-specific data from TreeBase
        self.allocate_data(self.n_nodes, n_features)        
        self._recursive_build(0, 0, n_samples)

    # Use cinit to initialize all arrays to empty: this prevents errors
    # in rare cases where __init__ is not called
    def __cinit__(self):
        self.data = np.empty((0, 0), dtype=DTYPE)
        self.idx_array = np.empty(0, dtype=ITYPE)
        self.node_data_arr = np.empty(0, dtype=NodeData)
        self.leaf_size = 0
        self.n_levels = 0
        self.n_nodes = 0

        # this should live in the derived class
        self.centroids_arr = np.empty((0, 0), dtype=DTYPE)

    def get_stats(self):
        return (self.n_trims, self.n_leaves, self.n_splits)

    # XXX: need to ensure picklability

    @cython.cdivision(True)
    cdef void _recursive_build(self, ITYPE_t i_node,
                               ITYPE_t idx_start, ITYPE_t idx_end):
        cdef ITYPE_t imax
        cdef ITYPE_t n_features = self.data.shape[1]
        cdef ITYPE_t n_points = idx_end - idx_start
        cdef ITYPE_t n_mid = n_points / 2

        # initialize node data
        self.init_node(i_node, idx_start, idx_end)

        if 2 * i_node + 1 >= self.n_nodes:
            self.node_data_arr[i_node].is_leaf = 1
            if idx_end - idx_start > 2 * self.leaf_size:
                # this shouldn't happen if our memory allocation is correct
                # we'll proactively prevent memory errors, but raise a warning
                # saying we're doing so.
                import warnings
                warnings.warn("Internal: memory layout is flawed: "
                              "not enough nodes allocated")

        elif idx_end - idx_start < 2:
            # this shouldn't happen if our memory allocation is correct
            # we'll proactively prevent memory errors, but raise a warning
            # saying we're doing so.
            import warnings
            warnings.warn("Internal: memory layout is flawed: "
                          "too many nodes allocated")
            self.node_data_arr[i_node].is_leaf = 1

        else:  # split node and recursively construct child nodes.
            self.node_data_arr[i_node].is_leaf = 0
            i_max = find_split_dim(self.data, self.idx_array,
                                   idx_start, idx_end)
            partition_indices(self.data, self.idx_array,
                              i_max, n_mid, idx_start, idx_end)

            self._recursive_build(2 * i_node + 1,
                                  idx_start, idx_start + n_mid)
            self._recursive_build(2 * i_node + 2,
                                  idx_start + n_mid, idx_end)

    #------------------------------------------------------------
    # below methods should be specialized for KD and Ball trees
    cdef void allocate_data(self, ITYPE_t n_nodes, ITYPE_t n_features):
        self.centroids_arr = np.zeros((n_nodes, n_features), dtype=DTYPE)

    cdef void init_node(self, ITYPE_t i_node,
                        ITYPE_t idx_start,
                        ITYPE_t idx_end):
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
        for i in range(idx_start, idx_end):
            radius = fmax(radius,
                          self.dm.dist(self.centroids_arr, i_node, 
                                       self.data, self.idx_array[i]))

        self.node_data_arr[i_node].radius = radius
        self.node_data_arr[i_node].idx_start = idx_start
        self.node_data_arr[i_node].idx_end = idx_end

    cdef inline DTYPE_t min_dist(self, ITYPE_t i_node,
                                 DTYPE_t[:, ::1] p, ITYPE_t i_p):
        cdef DTYPE_t dist_pt = self.dm.dist(p, i_p, self.centroids_arr, i_node)
        return fmax(0, dist_pt - self.node_data_arr[i_node].radius)
            
    cdef inline DTYPE_t max_dist(self, ITYPE_t i_node,
                                 DTYPE_t[:, ::1] p, ITYPE_t i_p):
        cdef DTYPE_t dist_pt = self.dm.dist(p, i_p, self.centroids_arr, i_node)
        return dist_pt + self.node_data_arr[i_node].radius

    #XXX need to implement rdist as well as dist

#!python
import numpy as np
from sklearn.utils import array2d

cimport numpy as np
cimport cython
from libc.math cimport fmax, fmin, fabs

from distmetrics cimport DistanceMetric
from distmetrics import choose_metric

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

cdef NodeData_t dummy
cdef NodeData_t[:] dummy_view = <NodeData_t[:1]> &dummy
NodeData = np.asarray(dummy_view).dtype

######################################################################
# BinaryTree class (Abstract Base Class)
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

    #------------------------------------------------------------
    # These keep track of statistics for building and querying
    cdef int n_trims
    cdef int n_leaves
    cdef int n_splits

    def __init__(self, X, leaf_size=20,
                 metric="minkowski", p=2, *args, **kwargs):
        raise ValueError("_BinaryTree cannot be instantiated on its own")

    # Use cinit to initialize all arrays to empty: this prevents errors
    # in rare cases where __init__ is not called
    def __cinit__(self):
        self.data = np.empty((0, 0), dtype=DTYPE)
        self.idx_array = np.empty((0, 0), dtype=ITYPE)
        self.node_data_arr = np.empty(0, dtype=NodeData)
        self.leaf_size = 0
        self.n_levels = 0
        self.n_nodes = 0

    #------------------------------------------------------------
    # The following methods should be overloaded by derived classes
    cdef void allocate_data(self, ITYPE_t n_nodes, ITYPE_t n_features):
        """Allocate data arrays needed for representation"""
        pass

    cdef void init_node(self, ITYPE_t i_node,
                        ITYPE_t idx_start, ITYPE_t idx_end):
        """Initialize the given node"""
        pass

    cdef DTYPE_t min_dist(self, ITYPE_t i_node,
                          DTYPE_t[:, ::1] pts, ITYPE_t i_pt):
        """Compute the minimum distance from a point to a node"""
        return 0.0

    cdef DTYPE_t min_rdist(self, ITYPE_t i_node,
                           DTYPE_t[:, ::1] pts, ITYPE_t i_pt):
        """Compute the minimum r-distance from a point to a node"""
        return 0.0

    #cdef DTYPE_t min_dist_dual(_BinaryTree bt1, ITYPE_t i_node1,
    #                           _BinaryTree bt2, ITYPE_t i_node2):
    #    """Compute the minimum distance between two nodes"""
    #    return 0.0

    #cdef DTYPE_t min_rdist_dual(_BinaryTree bt1, ITYPE_t i_node1,
    #                            _BinaryTree bt2, ITYPE_t i_node2):
    #    """Compute the minimum r-distance between two nodes"""
    #    return 0.0

    cdef void minmax_dist(self, ITYPE_t i_node, DTYPE_t[:, ::1] pts,
                          ITYPE_t i_pt, DTYPE_t* dmin, DTYPE_t* dmax):
        """Compute the min and max distance between a point and a node"""
        pass
    
    #------------------------------------------------------------

    def get_stats(self):
        return (self.n_trims, self.n_leaves, self.n_splits)

    def __init_common(self, X, leaf_size, metric, *args, **kwargs):
        """Common initialization steps"""
        # initialize distance metric
        self.dm = choose_metric(metric, *args, **kwargs)

        # validate data
        X = np.asarray(X, dtype=DTYPE, order='C')
        if X.size == 0:
            raise ValueError("X is an empty array")
        if X.ndim != 2:
            raise ValueError("X should have two dimensions")
        self.data = X

        # validate leaf size
        if leaf_size < 1:
            raise ValueError("leaf_size must be greater than or equal to 1")
        self.leaf_size = leaf_size
        
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
        self.node_data_arr = np.zeros(self.n_nodes, dtype=NodeData)

        # Allocate tree-specific data from TreeBase
        self.allocate_data(self.n_nodes, n_features)
        
        self._recursive_build(0, 0, n_samples)

    @cython.cdivision(True)
    cdef void _recursive_build(self, ITYPE_t i_node,
                               ITYPE_t idx_start, ITYPE_t idx_end):
        cdef ITYPE_t imax
        cdef ITYPE_t n_features = self.data.shape[1]
        cdef ITYPE_t n_points = idx_end - idx_start
        cdef ITYPE_t n_mid = n_points / 2

        # initialize node data
        self.init_node(self, i_node, idx_start, idx_end)

        # set up node info
        self.node_info[i_node].idx_start = idx_start
        self.node_info[i_node].idx_end = idx_end

        if 2 * i_node + 1 >= self.n_nodes:
            self.node_info[i_node].is_leaf = 1
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
            node_info.is_leaf = 1

        else:  # split node and recursively construct child nodes.
            # determine dimension on which to split
            self.node_info[i_node].is_leaf = 0

            i_max = find_split_dim(data, idx_array, n_features, n_points)

            # partition indices along this dimension
            partition_indices(data, idx_array, i_max, n_mid,
                              n_features, n_points)

            self._recursive_build(2 * i_node + 1,
                                  idx_start, idx_start + n_mid)
            self._recursive_build(2 * i_node + 2,
                                  idx_start + n_mid, idx_end)

    # XXX: need to make sure pickling will work
    #def __reduce__(self):
    #    """reduce method used for pickling"""
    #    return (newObj, (self.__class__,), self.__getstate__())

    #def __getstate__(self):
    #    """get state for pickling"""
    #    return (self.data,
    #            self.idx_array,
    #            self.node_data_arr1,
    #            self.node_data_arr2,
    #            self.node_data_arr,
    #            self.leaf_size,
    #            self.n_levels,
    #            self.n_nodes)

    #def __setstate__(self, state):
    #    """set state for pickling"""
    #    (self.data,
    #     self.idx_array,
    #     self.node_data_arr1,
    #     self.node_data_arr2,
    #     self.node_data_arr,
    #     self.leaf_size,
    #     self.n_levels,
    #     self.n_nodes) = state

    def query(self, X, k=1, return_distance=True, dualtree=False):
        """
        query(X, k=1, return_distance=True)

        query the Ball Tree for the k nearest neighbors

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
        cdef ITYPE_t n_neighbors = k
        cdef ITYPE_t n_features = self.data.shape[1]
        X = array2d(X, dtype=DTYPE, order='C')

        if X.shape[-1] != n_features:
            raise ValueError("query data dimension must match BallTree "
                             "data dimension")

        if self.data.shape[0] < n_neighbors:
            raise ValueError("k must be less than or equal "
                             "to the number of training points")

        # flatten X, and save original shape information
        orig_shape = X.shape
        X = X.reshape((-1, n_features))

        cdef ITYPE_t n_queries = X.shape[0]

        # allocate distances and indices for return
        cdef np.ndarray distances = np.zeros((X.shape[0], n_neighbors),
                                             dtype=DTYPE)
        distances.fill(INF)

        cdef np.ndarray idx_array = np.zeros((X.shape[0], n_neighbors),
                                             dtype=ITYPE)
        cdef np.ndarray Xarr = X

        # define some variables needed for the computation
        cdef np.ndarray bounds
        cdef ITYPE_t i
        cdef DTYPE_t* pt
        #cdef DTYPE_t* dist_ptr = <DTYPE_t*> distances.data
        cdef DTYPE_t* dist_ptr = <DTYPE_t*> np.PyArray_DATA(distances)
        #cdef ITYPE_t* idx_ptr = <ITYPE_t*> idx_array.data
        cdef ITYPE_t* idx_ptr = <ITYPE_t*> np.PyArray_DATA(idx_array)
        cdef DTYPE_t reduced_dist_LB

        # initialize heap
        heap_init(&self.heap, dist_ptr, idx_ptr, n_neighbors)

        self.n_trims = 0
        self.n_leaves = 0
        self.n_splits = 0

        if dualtree:
            raise ValueError("dual tree not supported")
            
            # build a tree on query data with the same metric as self
            #other = self.__class__(X, leaf_size=self.leaf_size)

            #reduced_dist_LB = min_rdist_dual(self, 0, other, 0)

            # bounds store the current furthest neighbor which is stored
            # in each node of the "other" tree.  This makes it so that we
            # don't need to repeatedly search every point in the node.
            #bounds = np.empty(other.data.shape[0])
            #bounds.fill(INF)

            #self.query_dual_(0, other, 0, n_neighbors,
            #                 dist_ptr, idx_ptr, reduced_dist_LB,
            #                 #<DTYPE_t*> bounds.data)
            #                <DTYPE_t*> np.PyArray_DATA(bounds))

        else:
            pt = <DTYPE_t*> np.PyArray_DATA(Xarr)
            #pt = <DTYPE_t*> Xarr.data
            for i in range(Xarr.shape[0]):
                reduced_dist_LB = min_rdist(self, 0, pt)
                self.query_one_(0, pt, n_neighbors,
                                dist_ptr, idx_ptr, reduced_dist_LB)

                dist_ptr += n_neighbors
                idx_ptr += n_neighbors
                pt += n_features

        dist_ptr = <DTYPE_t*> np.PyArray_DATA(distances)
        idx_ptr = <ITYPE_t*> np.PyArray_DATA(idx_array)
        #dist_ptr = <DTYPE_t*> distances.data
        #idx_ptr = <ITYPE_t*> idx_array.data
        for i in range(n_neighbors * n_queries):
            dist_ptr[i] = rdist_to_dist(dist_ptr[i])

        if heap_needs_final_sort(&self.heap):
            for i in range(n_queries):
                sort_dist_idx(dist_ptr, idx_ptr, n_neighbors)
                dist_ptr += n_neighbors
                idx_ptr += n_neighbors

        # deflatten results
        if return_distance:
            return (distances.reshape((orig_shape[:-1]) + (k,)),
                    idx_array.reshape((orig_shape[:-1]) + (k,)))
        else:
            return idx_array.reshape((orig_shape[:-1]) + (k,))

    
    cdef void query_one_(self,
                         ITYPE_t i_node,
                         DTYPE_t* pt,
                         ITYPE_t n_neighbors,
                         DTYPE_t* near_set_dist,
                         ITYPE_t* near_set_indx,
                         DTYPE_t reduced_dist_LB):
        cdef DTYPE_t* data = <DTYPE_t*> np.PyArray_DATA(self.data)
        cdef ITYPE_t* idx_array = <ITYPE_t*> np.PyArray_DATA(self.idx_array)
        cdef ITYPE_t n_features = self.data.shape[1]
        cdef NodeData* node_info = self.node_info(i_node)

        cdef DTYPE_t dist_pt, reduced_dist_LB_1, reduced_dist_LB_2
        cdef ITYPE_t i, i1, i2

        # Initialize the heap
        heap_init(&self.heap, near_set_dist, near_set_indx, n_neighbors)

        #------------------------------------------------------------
        # Case 1: query point is outside node radius:
        #         trim it from the query
        if reduced_dist_LB > heap_largest(&self.heap):
            self.n_trims += 1

        #------------------------------------------------------------
        # Case 2: this is a leaf node.  Update set of nearby points
        elif node_info.is_leaf:
            self.n_leaves += 1
            for i in range(node_info.idx_start, node_info.idx_end):
                dist_pt = rdist(pt, data + n_features * idx_array[i],
                                n_features)

                if dist_pt < heap_largest(&self.heap):
                    heap_insert(&self.heap, dist_pt, idx_array[i])

        #------------------------------------------------------------
        # Case 3: Node is not a leaf.  Recursively query subnodes
        #         starting with the closest
        else:
            self.n_splits += 1
            i1 = 2 * i_node + 1
            i2 = i1 + 1
            reduced_dist_LB_1 = min_rdist(self, i1, pt)
            reduced_dist_LB_2 = min_rdist(self, i2, pt)

            # recursively query subnodes
            if reduced_dist_LB_1 <= reduced_dist_LB_2:
                self.query_one_(i1, pt, n_neighbors, near_set_dist,
                                near_set_indx, reduced_dist_LB_1)
                self.query_one_(i2, pt, n_neighbors, near_set_dist,
                                near_set_indx, reduced_dist_LB_2)
            else:
                self.query_one_(i2, pt, n_neighbors, near_set_dist,
                                near_set_indx, reduced_dist_LB_2)
                self.query_one_(i1, pt, n_neighbors, near_set_dist,
                                near_set_indx, reduced_dist_LB_1)


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


######################################################################

cdef class BallTree(_BinaryTree):
    cdef DTYPE_t[:, ::1] centroids

    def __init__(self, X, leaf_size=20,
                 metric="minkowski", p=2, *args, **kwargs):
        self.__init_common(self, X, leaf_size, metric, *args, **kwargs)

    # Use cinit to initialize all arrays to empty: this prevents errors
    # in rare cases where __init__ is not called
    def __cinit__(self):
        self.centroids = np.empty((0, 0), dtype=DTYPE, order='C')

    cdef void allocate_data(self, ITYPE_t n_nodes, ITYPE_t n_features):
        self.centroids = np.zeros((n_nodes, n_features),
                                  dtype=DTYPE, order='C')

    @cython.cdivision(True)
    cdef void init_node(self, ITYPE_t i_node,
                        ITYPE_t idx_start, ITYPE_t idx_end):
        cdef ITYPE_t n_features = self.data.shape[1]
        cdef ITYPE_t n_points = idx_end - idx_start

        cdef DTYPE_t[::1] centroid = self.centroids[i_node]
        cdef DTYPE_t[::1] this_pt

        cdef ITYPE_t i, j
        cdef DTYPE_t radius
        cdef DTYPE_t *this_pt

        # determine Node centroid
        for j in range(n_features):
            centroid[j] = 0

        for i in range(idx_start, idx_end):
            this_pt = self.data[self.idx_array[i]]
            for j from 0 <= j < n_features:
                centroid[j] += this_pt[j]

        for j in range(n_features):
            centroid[j] /= n_points

        # determine Node radius
        radius = 0
        for i in range(idx_start, idx_end):
            radius = fmax(radius, self.dm.rdist(self.centroids, i_node,
                                                self.data, idx_array[i]))

        node_data.radius = self.dm.rdist_to_dist(radius)

    cdef DTYPE_t min_dist(self, ITYPE_t i_node,
                          DTYPE_t[:, ::1] pts, ITYPE_t i_pt):
        """Compute the minimum distance from a point to a node"""
        return fmax(0, self.dm.dist(self.centroids, i_node,
                                    pts, i_pt) - info.radius)

    cdef DTYPE_t min_rdist(self, ITYPE_t i_node,
                           DTYPE_t[:, ::1] pts, ITYPE_t i_pt):
        """Compute the minimum r-distance from a point to a node"""
        return dist_to_rdist(min_dist(self, i_node, pts, i_pt))

    #cdef DTYPE_t min_dist_dual(self, ITYPE_t i_node1,
    #                           BallTree other, ITYPE_t i_node2):
    #    """Compute the minimum distance between two nodes"""
    #    cdef ITYPE_t n_features = self.data.shape[1]
    #    cdef NodeData* info1 = self.node_data(i_node1)
    #    cdef NodeData* info2 = other.node_data(i_node2)
    #    cdef DTYPE_t* centroid1 = self.centroids(i_node1)
    #    cdef DTYPE_t* centroid2 = other.centroids(i_node2)
    #
    #    return fmax(0, (dist(centroid2, centroid1, n_features)
    #                    - info1.radius
    #                    - info2.radius))

    #cdef DTYPE_t min_rdist_dual(self, ITYPE_t i_node1,
    #                            BallTree other, ITYPE_t i_node2):
    #    """Compute the minimum r-distance between two nodes"""
    #    return dist_to_rdist(min_dist_dual(self, i_node1,
    #                                       other, i_node2))

    cdef void minmax_dist(self, ITYPE_t i_node, DTYPE_t[:, ::1] pts,
                          ITYPE_t i_pt, DTYPE_t* dmin, DTYPE_t* dmax):
        """Compute the min and max distance between a point and a node"""
        cdef DTYPE_t radius = self.node_data[i_node].radius
        cdef DTYPE_t dist_pt = self.dm.dist(self.centroids, i_node, pts, i_pt)

        dmin[0] = fmax(0, dist_pt - radius)
        dmax[0] = dist_pt + radius

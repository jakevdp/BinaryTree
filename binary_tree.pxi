from sklearn.utils import array2d

# explicitly define infinity
cdef DTYPE_t INF = np.inf

######################################################################
# newObj function
#  this is a helper function for pickling
def newObj(obj):
    return obj.__new__(obj)


######################################################################
# BinaryTree class.
# This is a base class for tree-based N-point queries
cdef class _BinaryTree(TreeBase):
    """Base class for KDTree and BallTree"""
    cdef np.ndarray data
    cdef np.ndarray idx_array
    cdef np.ndarray node_data_arr

    cdef ITYPE_t leaf_size
    cdef ITYPE_t n_levels
    cdef ITYPE_t n_nodes

    cdef HeapData heap

    # These keep track of statistics for building and querying
    cdef int n_trims
    cdef int n_leaves
    cdef int n_splits

    cdef NodeData* node_data(self, ITYPE_t i_node):
        return <NodeData*> np.PyArray_DATA(self.node_data_arr)

    # Use cinit to initialize all arrays to empty: this prevents errors
    # in rare cases where __init__ is not called
    def __cinit__(self):
        self.data = np.empty((0, 0))
        self.idx_array = np.empty((0, 0))
        self.node_data_arr = np.empty(0)
        self.leaf_size = 0
        self.n_levels = 0
        self.n_nodes = 0

    def get_stats(self):
        return (self.n_trims, self.n_leaves, self.n_splits)

    def __init__(self, X, leaf_size=20):
        raise ValueError("_BinaryTree cannot be instantiated on its own")

    def __init_common(self, X, leaf_size=20):
        """Common initialization steps"""
        self.data = np.asarray(X, dtype=DTYPE, order='C')

        if self.data.size == 0:
            raise ValueError("X is an empty array")

        if self.data.ndim != 2:
            raise ValueError("X should have two dimensions")

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
        self.node_data_arr = np.zeros(self.n_nodes * sizeof(NodeData),
                                      dtype='c', order='C')

        # Allocate tree-specific data from TreeBase
        self.allocate_data(self.n_nodes, n_features)
        self.init_data_views()
        
        self._recursive_build(0, 0, n_samples)

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
            # build a tree on query data with the same metric as self
            other = self.__class__(X, leaf_size=self.leaf_size)

            reduced_dist_LB = min_rdist_dual(self, 0, other, 0)

            # bounds store the current furthest neighbor which is stored
            # in each node of the "other" tree.  This makes it so that we
            # don't need to repeatedly search every point in the node.
            bounds = np.empty(other.data.shape[0])
            bounds.fill(INF)

            self.query_dual_(0, other, 0, n_neighbors,
                             dist_ptr, idx_ptr, reduced_dist_LB,
                             #<DTYPE_t*> bounds.data)
                            <DTYPE_t*> np.PyArray_DATA(bounds))

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

    def query_radius(self, X, r, return_distance=False,
                     int count_only=False, int sort_results=False):
        """
        query_radius(self, X, r, return_distance=False,
                     count_only = False, sort_results=False):

        query the Ball Tree for neighbors within a ball of size r

        Parameters
        ----------
        X : array-like, last dimension self.dim
            An array of points to query
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

        cdef np.ndarray idx_array, idx_array_i, distances, distances_i
        cdef np.ndarray pt, count
        cdef ITYPE_t count_i = 0
        cdef ITYPE_t n_features = self.data.shape[1]

        # prepare X for query
        X = array2d(X, dtype=DTYPE, order='C')
        if X.shape[-1] != self.data.shape[1]:
            raise ValueError("query data dimension must match BallTree "
                             "data dimension")

        # prepare r for query
        r = np.asarray(r, dtype=DTYPE, order='C')
        r = np.atleast_1d(r)
        if r.shape == (1,):
            r = r[0] * np.ones(X.shape[:-1], dtype=DTYPE)
        else:
            if r.shape != X.shape[:-1]:
                raise ValueError("r must be broadcastable to X.shape")

        # flatten X and r for iteration
        orig_shape = X.shape
        X = X.reshape((-1, X.shape[-1]))
        r = r.reshape(-1)
        
        cdef np.ndarray Xarr = X
        cdef np.ndarray rarr = r

        cdef DTYPE_t* Xdata = <DTYPE_t*> np.PyArray_DATA(Xarr)
        cdef DTYPE_t* rdata = <DTYPE_t*> np.PyArray_DATA(rarr)
        #cdef DTYPE_t* Xdata = <DTYPE_t*> Xarr.data
        #cdef DTYPE_t* rdata = <DTYPE_t*> rarr.data

        cdef ITYPE_t i

        # prepare variables for iteration
        if not count_only:
            idx_array = np.zeros(X.shape[0], dtype='object')
            if return_distance:
                distances = np.zeros(X.shape[0], dtype='object')

        idx_array_i = np.zeros(self.data.shape[0], dtype=ITYPE)
        distances_i = np.zeros(self.data.shape[0], dtype=DTYPE)
        count = np.zeros(X.shape[0], ITYPE)
        cdef ITYPE_t* count_data = <ITYPE_t*> np.PyArray_DATA(count)
        #cdef ITYPE_t* count_data = <ITYPE_t*> count.data

        #TODO: avoid enumerate and repeated allocation of pt slice
        for i in range(Xarr.shape[0]):
            count_data[i] = self.query_radius_one_(
                                      0,
                                      Xdata + i * n_features,
                                      rdata[i],
                                      <ITYPE_t*> np.PyArray_DATA(idx_array_i),
                                      <DTYPE_t*> np.PyArray_DATA(distances_i),
                                      #<ITYPE_t*> idx_array_i.data,
                                      #<DTYPE_t*> distances_i.data,
                                      0, count_only, return_distance)

            if count_only:
                pass
            else:
                if sort_results:
                    sort_dist_idx(#<DTYPE_t*> distances_i.data,
                                  #<ITYPE_t*> idx_array_i.data,
                                  <DTYPE_t*> np.PyArray_DATA(distances_i),
                                  <ITYPE_t*> np.PyArray_DATA(idx_array_i),
                                  count_data[i])

                idx_array[i] = idx_array_i[:count_data[i]].copy()
                if return_distance:
                    distances[i] = distances_i[:count_data[i]].copy()

        # deflatten results
        if count_only:
            return count.reshape(orig_shape[:-1])
        elif return_distance:
            return (idx_array.reshape(orig_shape[:-1]),
                    distances.reshape(orig_shape[:-1]))
        else:
            return idx_array.reshape(orig_shape[:-1])

    @cython.cdivision(True)
    cdef void _recursive_build(self, ITYPE_t i_node,
                               ITYPE_t idx_start, ITYPE_t idx_end):
        cdef ITYPE_t imax
        cdef ITYPE_t n_features = self.data.shape[1]
        cdef ITYPE_t n_points = idx_end - idx_start
        cdef ITYPE_t n_mid = n_points / 2
        cdef ITYPE_t* idx_array = (<ITYPE_t*> np.PyArray_DATA(self.idx_array)
                                   + idx_start)
        cdef DTYPE_t* data = <DTYPE_t*> np.PyArray_DATA(self.data)
        #cdef ITYPE_t* idx_array = (<ITYPE_t*> self.idx_array.data + idx_start)
        #cdef DTYPE_t* data = <DTYPE_t*> self.data.data

        # initialize node data
        cdef NodeData* node_info = init_node(self, i_node, idx_start, idx_end)

        # set up node info
        node_info.idx_start = idx_start
        node_info.idx_end = idx_end

        if 2 * i_node + 1 >= self.n_nodes:
            node_info.is_leaf = 1
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
            node_info.is_leaf = 0

            i_max = find_split_dim(data, idx_array, n_features, n_points)

            # partition indices along this dimension
            partition_indices(data, idx_array, i_max, n_mid,
                              n_features, n_points)

            self._recursive_build(2 * i_node + 1,
                                  idx_start, idx_start + n_mid)
            self._recursive_build(2 * i_node + 2,
                                  idx_start + n_mid, idx_end)

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

    cdef void query_dual_(self,
                          ITYPE_t i_node1,
                          _BinaryTree other,
                          ITYPE_t i_node2,
                          ITYPE_t n_neighbors,
                          DTYPE_t* near_set_dist,
                          ITYPE_t* near_set_indx,
                          DTYPE_t reduced_dist_LB,
                          DTYPE_t* bounds):
        cdef ITYPE_t n_features = self.data.shape[1]

        cdef NodeData* node_info1 = self.node_info(i_node1)
        cdef NodeData* node_info2 = other.node_info(i_node2)
        
        #cdef DTYPE_t* data1 = <DTYPE_t*> self.data.data
        #cdef DTYPE_t* data2 = <DTYPE_t*> other.data.data
        cdef DTYPE_t* data1 = <DTYPE_t*> np.PyArray_DATA(self.data)
        cdef DTYPE_t* data2 = <DTYPE_t*> np.PyArray_DATA(other.data)

        #cdef ITYPE_t* idx_array1 = <ITYPE_t*> self.idx_array.data
        #cdef ITYPE_t* idx_array2 = <ITYPE_t*> other.idx_array.data
        cdef ITYPE_t* idx_array1 = <ITYPE_t*> np.PyArray_DATA(self.idx_array)
        cdef ITYPE_t* idx_array2 = <ITYPE_t*> np.PyArray_DATA(other.idx_array)

        cdef DTYPE_t dist_pt, reduced_dist_LB1, reduced_dist_LB2
        cdef ITYPE_t i1, i2

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
                heap_init(&self.heap,
                          near_set_dist + idx_array2[i2] * n_neighbors,
                          near_set_indx + idx_array2[i2] * n_neighbors,
                          n_neighbors)

                if heap_largest(&self.heap) <= reduced_dist_LB:
                    continue

                for i1 in range(node_info1.idx_start, node_info1.idx_end):
                    dist_pt = rdist(data1 + n_features * idx_array1[i1],
                                    data2 + n_features * idx_array2[i2],
                                    n_features)
                    if dist_pt < heap_largest(&self.heap):
                        heap_insert(&self.heap, dist_pt, idx_array1[i1])
                
                # keep track of node bound
                bounds[i_node2] = fmax(bounds[i_node2],
                                       heap_largest(&self.heap))
            
        #------------------------------------------------------------
        # Case 3a: node 1 is a leaf: split node 2 and recursively
        #          query, starting with the nearest node
        elif node_info1.is_leaf:
            reduced_dist_LB1 = min_rdist_dual(self, i_node1,
                                              other, 2 * i_node2 + 1)
            reduced_dist_LB2 = min_rdist_dual(self, i_node1,
                                              other, 2 * i_node2 + 2)

            if reduced_dist_LB1 < reduced_dist_LB2:
                self.query_dual_(i_node1, other, 2 * i_node2 + 1, n_neighbors,
                                 near_set_dist, near_set_indx,
                                 reduced_dist_LB1, bounds)
                self.query_dual_(i_node1, other, 2 * i_node2 + 2, n_neighbors,
                                 near_set_dist, near_set_indx,
                                 reduced_dist_LB2, bounds)
            else:
                self.query_dual_(i_node1, other, 2 * i_node2 + 2, n_neighbors,
                                 near_set_dist, near_set_indx,
                                 reduced_dist_LB2, bounds)
                self.query_dual_(i_node1, other, 2 * i_node2 + 1, n_neighbors,
                                 near_set_dist, near_set_indx,
                                 reduced_dist_LB1, bounds)
            
            # update node bound information
            bounds[i_node2] = fmax(bounds[2 * i_node2 + 1],
                                   bounds[2 * i_node2 + 2])
            
        #------------------------------------------------------------
        # Case 3b: node 2 is a leaf: split node 1 and recursively
        #          query, starting with the nearest node
        elif node_info2.is_leaf:
            reduced_dist_LB1 = min_rdist_dual(self, 2 * i_node1 + 1,
                                              other, i_node2)
            reduced_dist_LB2 = min_rdist_dual(self, 2 * i_node1 + 2,
                                              other, i_node2)

            if reduced_dist_LB1 < reduced_dist_LB2:
                self.query_dual_(2 * i_node1 + 1, other, i_node2, n_neighbors,
                                 near_set_dist, near_set_indx,
                                 reduced_dist_LB1, bounds)
                self.query_dual_(2 * i_node1 + 2, other, i_node2, n_neighbors,
                                 near_set_dist, near_set_indx,
                                 reduced_dist_LB2, bounds)
            else:
                self.query_dual_(2 * i_node1 + 2, other, i_node2, n_neighbors,
                                 near_set_dist, near_set_indx,
                                 reduced_dist_LB2, bounds)
                self.query_dual_(2 * i_node1 + 1, other, i_node2, n_neighbors,
                                 near_set_dist, near_set_indx,
                                 reduced_dist_LB1, bounds)
        
        #------------------------------------------------------------
        # Case 4: neither node is a leaf:
        #         split both and recursively query all four pairs
        else:
            reduced_dist_LB1 = min_rdist_dual(self, 2 * i_node1 + 1,
                                              other, 2 * i_node2 + 1)
            reduced_dist_LB2 = min_rdist_dual(self, 2 * i_node1 + 2,
                                              other, 2 * i_node2 + 1)

            if reduced_dist_LB1 < reduced_dist_LB2:
                self.query_dual_(2 * i_node1 + 1, other, 2 * i_node2 + 1,
                                 n_neighbors, near_set_dist, near_set_indx,
                                 reduced_dist_LB1, bounds)
                self.query_dual_(2 * i_node1 + 2, other, 2 * i_node2 + 1,
                                 n_neighbors, near_set_dist, near_set_indx,
                                 reduced_dist_LB2, bounds)
            else:
                self.query_dual_(2 * i_node1 + 2, other, 2 * i_node2 + 1,
                                 n_neighbors, near_set_dist, near_set_indx,
                                 reduced_dist_LB2, bounds)
                self.query_dual_(2 * i_node1 + 1, other, 2 * i_node2 + 1,
                                 n_neighbors, near_set_dist, near_set_indx,
                                 reduced_dist_LB1, bounds)

            reduced_dist_LB1 = min_rdist_dual(self, 2 * i_node1 + 1,
                                              other, 2 * i_node2 + 2)
            reduced_dist_LB2 = min_rdist_dual(self, 2 * i_node1 + 2,
                                              other, 2 * i_node2 + 2)
            if reduced_dist_LB1 < reduced_dist_LB2:
                self.query_dual_(2 * i_node1 + 1, other, 2 * i_node2 + 2,
                                 n_neighbors, near_set_dist, near_set_indx,
                                 reduced_dist_LB1, bounds)
                self.query_dual_(2 * i_node1 + 2, other, 2 * i_node2 + 2,
                                 n_neighbors, near_set_dist, near_set_indx,
                                 reduced_dist_LB2, bounds)
            else:
                self.query_dual_(2 * i_node1 + 2, other, 2 * i_node2 + 2,
                                 n_neighbors, near_set_dist, near_set_indx,
                                 reduced_dist_LB2, bounds)
                self.query_dual_(2 * i_node1 + 1, other, 2 * i_node2 + 2,
                                 n_neighbors, near_set_dist, near_set_indx,
                                 reduced_dist_LB1, bounds)
            
            # update node bound information
            bounds[i_node2] = fmax(bounds[2 * i_node2 + 1],
                                   bounds[2 * i_node2 + 2])

    cdef ITYPE_t query_radius_one_(self,
                                   ITYPE_t i_node,
                                   DTYPE_t* pt, DTYPE_t r,
                                   ITYPE_t* indices,
                                   DTYPE_t* distances,
                                   ITYPE_t count,
                                   int count_only,
                                   int return_distance):
        #cdef DTYPE_t* data = <DTYPE_t*> self.data.data
        #cdef ITYPE_t* idx_array = <ITYPE_t*> self.idx_array.data
        cdef DTYPE_t* data = <DTYPE_t*> np.PyArray_DATA(self.data)
        cdef ITYPE_t* idx_array = <ITYPE_t*> np.PyArray_DATA(self.idx_array)
        cdef ITYPE_t n_features = self.data.shape[1]

        cdef NodeData* node_info = self.node_info(i_node)

        cdef ITYPE_t i
        cdef DTYPE_t reduced_r

        cdef DTYPE_t dist_pt, dist_LB, dist_UB
        minmax_dist(self, i_node, pt, &dist_LB, &dist_UB)

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
                        raise ValueError("count too big")
                    indices[count] = idx_array[i]
                    if return_distance:
                        distances[count] = dist(pt, (data + n_features
                                                     * idx_array[i]),
                                                n_features)
                    count += 1

        #------------------------------------------------------------
        # Case 3: this is a leaf node.  Go through all points to
        #         determine if they fall within radius
        elif node_info.is_leaf:
            reduced_r = dist_to_rdist(r)

            for i in range(node_info.idx_start, node_info.idx_end):
                dist_pt = rdist(pt, (data + n_features
                                     * idx_array[i]),
                                n_features)
                if dist_pt <= reduced_r:
                    if (count < 0) or (count >= self.data.shape[0]):
                        raise ValueError("Fatal: count out of range")
                    if count_only:
                        pass
                    else:
                        indices[count] = idx_array[i]
                        if return_distance:
                            distances[count] = rdist_to_dist(dist_pt)
                    count += 1

        #------------------------------------------------------------
        # Case 4: Node is not a leaf.  Recursively query subnodes
        else:
            count = self.query_radius_one_(2 * i_node + 1, pt, r,
                                           indices, distances, count,
                                           count_only, return_distance)
            count = self.query_radius_one_(2 * i_node + 2, pt, r,
                                           indices, distances, count,
                                           count_only, return_distance)

        return count

    cdef NodeData* node_info(self, ITYPE_t i_node):
        return <NodeData*> np.PyArray_DATA(self.node_data_arr) + i_node


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


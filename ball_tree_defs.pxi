######################################################################
# NodeData struct
#  used to keep track of information for individual nodes
cdef struct NodeData:
    ITYPE_t idx_start
    ITYPE_t idx_end
    ITYPE_t is_leaf
    DTYPE_t radius


######################################################################
# TreeBase class
#  used to keep track of information for building the tree.
#  _BinaryTree will inherit from this to allow it to implement
#  different kinds of binary trees.
cdef class TreeBase:
    cdef np.ndarray centroids_arr
    #cdef DTYPE_t[:, ::1] centroids

    # Use cinit to initialize all arrays to empty: this prevents errors
    # in rare cases where __init__ is not called
    def __cinit__(self):
        self.centroids_arr = np.empty((0, 0), dtype=DTYPE, order='C')
        #self.centroids = self.centroids_arr

    cdef void allocate_data(self, n_nodes, n_features):
        self.centroids_arr = np.zeros((n_nodes, n_features),
                                      dtype=DTYPE, order='C')

    cdef void init_data_views(self):
        pass
        #self.centroids = self.centroids_arr

    cdef DTYPE_t* centroids(self, ITYPE_t i_node):
        return (<DTYPE_t*> np.PyArray_DATA(self.centroids_arr)
                + i_node * self.centroids_arr.shape[1])

@cython.cdivision(True)
cdef NodeData* init_node(_BinaryTree bt, ITYPE_t i_node,
                         ITYPE_t idx_start, ITYPE_t idx_end):
    cdef ITYPE_t n_features = bt.data.shape[1]
    cdef ITYPE_t n_points = idx_end - idx_start

    cdef ITYPE_t* idx_array = <ITYPE_t*> np.PyArray_DATA(bt.idx_array)
    cdef DTYPE_t* data = <DTYPE_t*> np.PyArray_DATA(bt.data)
    cdef NodeData* node_data = bt.node_data(i_node)
    cdef DTYPE_t* centroid = bt.centroids(i_node)

    cdef ITYPE_t i, j
    cdef DTYPE_t radius
    cdef DTYPE_t *this_pt

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
        radius = fmax(radius, rdist(centroid,
                                    data + n_features * idx_array[i],
                                    n_features))

    node_data.radius = rdist_to_dist(radius)

    return node_data


cdef inline DTYPE_t min_dist(_BinaryTree bt, ITYPE_t i_node, DTYPE_t* pt):
    cdef ITYPE_t n_features = bt.data.shape[1]
    cdef NodeData* info = bt.node_data(i_node)
    cdef DTYPE_t* centroid = bt.centroids(i_node)

    return fmax(0, dist(pt, centroid, n_features) - info.radius)


cdef inline DTYPE_t min_rdist(_BinaryTree bt, ITYPE_t i_node, DTYPE_t* pt):
    return dist_to_rdist(min_dist(bt, i_node, pt))


cdef inline DTYPE_t min_dist_dual(_BinaryTree bt1, ITYPE_t i_node1,
                                  _BinaryTree bt2, ITYPE_t i_node2):
    cdef ITYPE_t n_features = bt1.data.shape[1]
    cdef NodeData* info1 = bt1.node_data(i_node1)
    cdef NodeData* info2 = bt2.node_data(i_node2)
    cdef DTYPE_t* centroid1 = bt1.centroids(i_node1)
    cdef DTYPE_t* centroid2 = bt2.centroids(i_node2)

    return fmax(0, (dist(centroid2, centroid1, n_features)
                    - info1.radius
                    - info2.radius))


cdef inline DTYPE_t min_rdist_dual(_BinaryTree bt1, ITYPE_t i_node1,
                                   _BinaryTree bt2, ITYPE_t i_node2):
    return dist_to_rdist(min_dist_dual(bt1, i_node1,
                                       bt2, i_node2))


cdef inline void minmax_dist(_BinaryTree bt, ITYPE_t i_node, DTYPE_t* pt,
                             DTYPE_t* dmin, DTYPE_t* dmax):
    cdef ITYPE_t n_features = bt.data.shape[1]
    cdef NodeData* info = bt.node_data(i_node)
    cdef DTYPE_t* centroid = bt.centroids(i_node)
    cdef DTYPE_t dist_pt = dist(pt, centroid, n_features)

    dmin[0] = fmax(0, dist_pt - info.radius)
    dmax[0] = dist_pt + info.radius

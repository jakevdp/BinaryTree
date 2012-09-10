######################################################################
# NodeData struct
#  used to keep track of information for individual nodes
cdef struct NodeData:
    ITYPE_t idx_start
    ITYPE_t idx_end
    ITYPE_t is_leaf


######################################################################
# TreeBase class
#  used to keep track of information for building the tree.
#  _BinaryTree will inherit from this to allow it to implement
#  different kinds of binary trees.
cdef class TreeBase:
    cdef np.ndarray lower_bounds_arr
    cdef np.ndarray upper_bounds_arr
    cdef DTYPE_t[:, ::1] lower_bounds
    cdef DTYPE_t[:, ::1] upper_bounds

    # Use cinit to initialize all arrays to empty: this prevents errors
    # in rare cases where __init__ is not called
    def __cinit__(self):
        self.lower_bounds_arr = np.zeros((0, 0), dtype=DTYPE, order='C')
        self.upper_bounds_arr = np.zeros((0, 0), dtype=DTYPE, order='C')
        self.lower_bounds = self.lower_bounds_arr
        self.upper_bounds = self.upper_bounds_arr

    cdef void allocate_data(self, n_nodes, n_features):
        self.lower_bounds_arr = np.zeros((n_nodes, n_features),
                                         dtype='c', order='C')
        self.upper_bounds_arr = np.zeros((n_nodes, n_features),
                                         dtype='c', order='C')

    cdef void init_data_views(self):
        self.lower_bounds = self.lower_bounds_arr
        self.upper_bounds = self.upper_bounds_arr


@cython.cdivision(True)
cdef NodeData* init_node(_BinaryTree bt, ITYPE_t i_node,
                         ITYPE_t idx_start, ITYPE_t idx_end):
    cdef ITYPE_t n_features = bt.data.shape[1]
    cdef ITYPE_t n_points = idx_end - idx_start

    #cdef ITYPE_t* idx_array = <ITYPE_t*> bt.idx_array.data
    #cdef DTYPE_t* data = <DTYPE_t*> bt.data.data
    cdef ITYPE_t* idx_array = <ITYPE_t*> np.PyArray_DATA(bt.idx_array)
    cdef DTYPE_t* data = <DTYPE_t*> np.PyArray_DATA(bt.data)
    cdef NodeData* node_data = bt.node_data(i_node)
    cdef DTYPE_t* lower = bt.lower_bounds(i_node)
    cdef DTYPE_t* upper = bt.upper_bounds(i_node)

    cdef ITYPE_t i, j
    cdef DTYPE_t *this_pt

    # determine Node bounds
    for j in range(n_features):
        lower[j] = INF
        upper[j] = -INF

    for i in range(idx_start, idx_end):
        this_pt = data + n_features * idx_array[i]
        for j in range(n_features):
            lower[j] = fmin(lower[j], this_pt[j])
            upper[j] = fmax(upper[j], this_pt[j])

    return node_data


@cython.cdivision(True)
cdef DTYPE_t min_rdist(_BinaryTree bt, ITYPE_t i_node, DTYPE_t* pt):
    cdef ITYPE_t n_features = bt.data.shape[1]
    cdef DTYPE_t* lower = bt.lower_bounds(i_node)
    cdef DTYPE_t* upper = bt.upper_bounds(i_node)

    cdef DTYPE_t d, d_lo, d_hi, rdist=0.0
    cdef ITYPE_t j

    # here we'll use the fact that x + abs(x) = 2 * max(x, 0)
    for j in range(n_features):
        d_lo = lower[j] - pt[j]
        d_hi = pt[j] - upper[j]
        d = (d_lo + fabs(d_lo)) + (d_hi + fabs(d_hi))

        #rdist += d ^ p
        rdist += dist_to_rdist(d)

    #rdist /= 2 ^ p
    return rdist / dist_to_rdist(2.0)

cdef DTYPE_t min_dist(_BinaryTree bt, ITYPE_t i_node, DTYPE_t* pt):
    return rdist_to_dist(min_rdist(bt, i_node, pt))


@cython.cdivision(True)
cdef DTYPE_t min_rdist_dual(_BinaryTree bt1, ITYPE_t i_node1,
                            _BinaryTree bt2, ITYPE_t i_node2):
    cdef ITYPE_t n_features = bt1.data.shape[1]

    cdef DTYPE_t* lower1 = bt1.lower_bounds(i_node1)
    cdef DTYPE_t* upper1 = bt1.upper_bounds(i_node1)

    cdef DTYPE_t* lower2 = bt2.lower_bounds(i_node2)
    cdef DTYPE_t* upper2 = bt2.upper_bounds(i_node2)

    cdef DTYPE_t d, d1, d2, rdist=0.0
    cdef DTYPE_t zero = 0.0
    cdef ITYPE_t j

    # here we'll use the fact that x + abs(x) = 2 * max(x, 0)
    for j in range(n_features):
        d1 = lower1[j] - upper2[j]
        d2 = lower2[j] - upper1[j]
        d = (d1 + fabs(d1)) + (d2 + fabs(d2))

        #rdist += d ^ p
        rdist += dist_to_rdist(d)

    #rdist /= 2 ^ p
    return rdist / dist_to_rdist(2.0)


cdef DTYPE_t min_dist_dual(_BinaryTree bt1, ITYPE_t i_node1,
                           _BinaryTree bt2, ITYPE_t i_node2):
    return rdist_to_dist(min_rdist_dual(bt1, i_node1,
                                        bt2, i_node2))

#!python

import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport fmax, fmin, fabs

cdef class BallTree(_BinaryTree):
    cdef DTYPE_t[:, ::1] centroids

    def __init__(self, X, leaf_size=20,
                 metric="minkowski", p=2, *args, **kwargs):
        self.dm = c

    # Use cinit to initialize all arrays to empty: this prevents errors
    # in rare cases where __init__ is not called
    def __cinit__(self):
        self.centroids = np.empty((0, 0), dtype=DTYPE, order='C')

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
                                                self.data, idx_array[i])

        node_data.radius = self.dm.rdist_to_dist(radius)

    cdef void allocate_data(self, ITYPE_t n_nodes, ITYPE_t n_features):
        self.centroids = np.zeros((n_nodes, n_features),
                                  dtype=DTYPE, order='C')

    cdef DTYPE_t min_dist(self, ITYPE_t i_node, DTYPE_t* pt):
        """Compute the minimum distance from a point to a node"""
        cdef ITYPE_t n_features = self.data.shape[1]
        cdef NodeData_t* info = self.node_data(i_node)
        cdef DTYPE_t* centroid = self.centroids(i_node)

        return fmax(0, dist(pt, centroid, n_features) - info.radius)

    cdef DTYPE_t min_rdist(self, ITYPE_t i_node, DTYPE_t* pt):
        """Compute the minimum r-distance from a point to a node"""
        return dist_to_rdist(min_dist(self, i_node, pt))

    cdef DTYPE_t min_dist_dual(self, ITYPE_t i_node1,
                               BallTree other, ITYPE_t i_node2):
        """Compute the minimum distance between two nodes"""
        cdef ITYPE_t n_features = self.data.shape[1]
        cdef NodeData* info1 = self.node_data(i_node1)
        cdef NodeData* info2 = other.node_data(i_node2)
        cdef DTYPE_t* centroid1 = self.centroids(i_node1)
        cdef DTYPE_t* centroid2 = other.centroids(i_node2)

        return fmax(0, (dist(centroid2, centroid1, n_features)
                        - info1.radius
                        - info2.radius))

    cdef DTYPE_t min_rdist_dual(self, ITYPE_t i_node1,
                                BallTree other, ITYPE_t i_node2):
        """Compute the minimum r-distance between two nodes"""
        return dist_to_rdist(min_dist_dual(self, i_node1,
                                           other, i_node2))

    cdef void minmax_dist(self, ITYPE_t i_node, DTYPE_t* pt,
                          DTYPE_t* dmin, DTYPE_t* dmax):
        """Compute the min and max distance between a point and a node"""
        cdef ITYPE_t n_features = self.data.shape[1]
        cdef NodeData* info = self.node_data(i_node)
        cdef DTYPE_t* centroid = self.centroids(i_node)
        cdef DTYPE_t dist_pt = dist(pt, centroid, n_features)

        dmin[0] = fmax(0, dist_pt - info.radius)
        dmax[0] = dist_pt + info.radius

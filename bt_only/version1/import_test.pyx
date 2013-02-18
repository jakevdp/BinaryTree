#!python
import numpy as np
from dist_metrics cimport DistanceMetric
from dist_metrics import EuclideanDistance

from typedefs import DTYPE
from typedefs cimport DTYPE_t


def euclidean_pairwise_class(DTYPE_t[:, ::1] X, DTYPE_t[:, ::1] Y):
    cdef DistanceMetric eucl_dist = EuclideanDistance()

    assert X.shape[1] == Y.shape[1]
    cdef DTYPE_t[:, ::1] D = np.zeros((X.shape[0], Y.shape[0]), dtype=DTYPE)

    for i1 in range(X.shape[0]):
        for i2 in range(Y.shape[0]):
            D[i1, i2] = eucl_dist.dist(X, i1, Y, i2)
    return np.asarray(D)

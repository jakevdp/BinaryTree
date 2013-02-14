import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.distance import pdist, cdist, squareform
from version2.dist_metrics import EuclideanDistance


def test_euclidean_cdist(N=100, D=5):
    X = np.random.random((N, D))
    Y = np.random.random((N, D))
    D1 = cdist(X, Y)
    D2 = EuclideanDistance().pairwise(X, Y)

    assert_allclose(D1, D2)


def test_euclidean_pdist(N=100, D=5):
    X = np.random.random((N, D))
    D1 = squareform(pdist(X))
    D2 = EuclideanDistance().pairwise(X)

    assert_allclose(D1, D2)

import numpy as np
from numpy.testing import assert_allclose
from binary_tree import BallTree, KDTree
from distmetrics import Distance

METRICS = {'euclidean':{},
           'cityblock':{},
           'chebyshev':{},
           'minkowski':{'p':3},
           'minkowski':{'p':4}}

AXIS_ALIGNED_METRICS = ['euclidean', 'cityblock', 'minkowski']


def compute_neighbors_brute(X, Y, k, metric, **kwargs):
    D = Distance(metric, **kwargs).pairwise(Y, X)
    ind = np.argsort(D, axis=1)[:, :k]
    dist = D[np.arange(Y.shape[0])[:, None], ind]
    return dist, ind


def test_ball_tree():
    X = np.random.random((40, 3))
    Y = np.random.random((10, 3))

    def check_neighbors(dualtree, k, metric, kwargs):
        bt = BallTree(X, leaf_size=1, metric=metric, **kwargs)
        dist1, ind1 = bt.query(Y, k, dualtree=dualtree)
        dist2, ind2 = compute_neighbors_brute(X, Y, k, metric, **kwargs)

        assert_allclose(dist1, dist2)
        assert_allclose(ind1, ind2)

    for (metric, kwargs) in METRICS.iteritems():
        for k in (1, 3, 5):
            for dualtree in (True, False):
                yield (check_neighbors, dualtree, k, metric, kwargs)


def test_kd_tree():
    X = np.random.random((40, 3))
    Y = np.random.random((10, 3))

    def check_neighbors(dualtree, k, metric, kwargs):
        kdt = KDTree(X, leaf_size=1, metric=metric, **kwargs)
        dist1, ind1 = kdt.query(Y, k, dualtree=dualtree)
        dist2, ind2 = compute_neighbors_brute(X, Y, k, metric, **kwargs)

        assert_allclose(dist1, dist2)
        assert_allclose(ind1, ind2)

    for (metric, kwargs) in METRICS.iteritems():
        if metric not in AXIS_ALIGNED_METRICS:
            continue
        for k in (1, 3, 5):
            for dualtree in (True, False):
                if dualtree is True:
                    continue
                yield (check_neighbors, dualtree, k, metric, kwargs)


if __name__ == '__main__':
    import nose
    nose.runmodule()

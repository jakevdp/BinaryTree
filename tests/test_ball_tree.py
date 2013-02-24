import numpy as np
from numpy.testing import assert_allclose
from ball_tree import BallTree, DistanceMetric

METRICS = {'euclidean':{},
           'manhattan':{},
           'minkowski':{'p':3}}


def brute_force_neighbors(X, Y, k, metric, **kwargs):
    D = DistanceMetric.get_metric(metric, **kwargs).pairwise(Y, X)
    ind = np.argsort(D, axis=1)[:, :k]
    dist = D[np.arange(Y.shape[0])[:, None], ind]
    return dist, ind


def test_ball_tree():
    X = np.random.random((40, 3))
    Y = np.random.random((10, 3))

    def check_neighbors(dualtree, breadth_first, k, metric, kwargs):
        bt = BallTree(X, leaf_size=1, metric=metric, **kwargs)
        dist1, ind1 = bt.query(Y, k, dualtree=dualtree,
                               breadth_first=breadth_first)
        dist2, ind2 = brute_force_neighbors(X, Y, k, metric, **kwargs)

        assert_allclose(ind1, ind2)
        assert_allclose(dist1, dist2)

    for (metric, kwargs) in METRICS.iteritems():
        for k in (1, 3, 5):
            for dualtree in (True, False):
                for breadth_first in (True, False):
                    yield (check_neighbors,
                           dualtree, breadth_first,
                           k, metric, kwargs)


if __name__ == '__main__':
    import nose
    nose.runmodule()

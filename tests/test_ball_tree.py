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


def test_ball_tree_query():
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


def test_ball_tree_query_radius(n_samples=100, n_features=10):
    X = 2 * np.random.random(size=(n_samples, n_features)) - 1
    query_pt = np.zeros(n_features, dtype=float)

    eps = 1E-15  # roundoff error can cause test to fail
    bt = BallTree(X, leaf_size=5)
    rad = np.sqrt(((X - query_pt) ** 2).sum(1))

    for r in np.linspace(rad[0], rad[-1], 100):
        ind = bt.query_radius(query_pt, r + eps)[0]
        i = np.where(rad <= r + eps)[0]

        ind.sort()
        i.sort()

        assert_allclose(i, ind)


def test_ball_tree_query_radius_distance(n_samples=100, n_features=10):
    X = 2 * np.random.random(size=(n_samples, n_features)) - 1
    query_pt = np.zeros(n_features, dtype=float)

    eps = 1E-15  # roundoff error can cause test to fail
    bt = BallTree(X, leaf_size=5)
    rad = np.sqrt(((X - query_pt) ** 2).sum(1))

    for r in np.linspace(rad[0], rad[-1], 100):
        ind, dist = bt.query_radius(query_pt, r + eps, return_distance=True)

        ind = ind[0]
        dist = dist[0]

        d = np.sqrt(((query_pt - X[ind]) ** 2).sum(1))

        assert_allclose(d, dist)

def test_ball_tree_KDE(n_samples=100, n_features=3):
    X = np.random.random((n_samples, n_features))
    bt = BallTree(X, leaf_size=10)

    for h in [0.001, 0.01, 0.1, 1.0]:
        d = X[:, None, :] - X
        dens_true = np.exp(-0.5 * (d ** 2).sum(-1) / h ** 2).sum(-1)
        def check_results(h, atol):
            dens = bt.kernel_density(X, h, atol=atol)
            assert_allclose(dens, dens_true, atol=atol, rtol=1E-10)

        for atol in [0, 1E-5, 0.1]:
            yield check_results, h, atol


if __name__ == '__main__':
    import nose
    nose.runmodule()

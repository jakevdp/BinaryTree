import numpy as np
from numpy.testing import assert_allclose
from ball_tree import BallTree, DistanceMetric

V = np.random.random((3, 3))
V = np.dot(V, V.T)

DIMENSION = 3

METRICS = {'euclidean':{},
           'manhattan':{},
           'minkowski':dict(p=3),
           'chebyshev':{},
           'seuclidean':dict(V=np.random.random(DIMENSION)),
           'wminkowski':dict(p=3, w=np.random.random(DIMENSION)),
           'mahalanobis':dict(V=V)}

DISCRETE_METRICS = ['hamming',
                    'canberra',
                    'braycurtis']

BOOLEAN_METRICS = ['matching', 'jaccard', 'dice', 'kulsinski',
                   'rogerstanimoto', 'russellrao', 'sokalmichener',
                   'sokalsneath']


def brute_force_neighbors(X, Y, k, metric, **kwargs):
    D = DistanceMetric.get_metric(metric, **kwargs).pairwise(Y, X)
    ind = np.argsort(D, axis=1)[:, :k]
    dist = D[np.arange(Y.shape[0])[:, None], ind]
    return dist, ind


def test_ball_tree_query():
    np.random.seed(0)
    X = np.random.random((40, DIMENSION))
    Y = np.random.random((10, DIMENSION))

    def check_neighbors(dualtree, breadth_first, k, metric, kwargs):
        bt = BallTree(X, leaf_size=1, metric=metric, **kwargs)
        dist1, ind1 = bt.query(Y, k, dualtree=dualtree,
                               breadth_first=breadth_first)
        dist2, ind2 = brute_force_neighbors(X, Y, k, metric, **kwargs)

        # don't check indices here: if there are any duplicate distances,
        # the indices may not match.  Distances should not have this problem.
        assert_allclose(dist1, dist2)

    for (metric, kwargs) in METRICS.iteritems():
        for k in (1, 3, 5):
            for dualtree in (True, False):
                for breadth_first in (True, False):
                    yield (check_neighbors,
                           dualtree, breadth_first,
                           k, metric, kwargs)


def test_ball_tree_query_boolean_metrics():
    np.random.seed(0)
    X = np.random.random((40, 10)).round(0)
    Y = np.random.random((10, 10)).round(0)
    k = 5

    def check_neighbors(metric):
        bt = BallTree(X, leaf_size=1, metric=metric)
        dist1, ind1 = bt.query(Y, k)
        dist2, ind2 = brute_force_neighbors(X, Y, k, metric)
        assert_allclose(dist1, dist2)

    for metric in BOOLEAN_METRICS:
        yield check_neighbors, metric


def test_ball_tree_query_discrete_metrics():
    np.random.seed(0)
    X = (4 * np.random.random((40, 10))).round(0)
    Y = (4 * np.random.random((10, 10))).round(0)
    k = 5

    def check_neighbors(metric):
        bt = BallTree(X, leaf_size=1, metric=metric)
        dist1, ind1 = bt.query(Y, k)
        dist2, ind2 = brute_force_neighbors(X, Y, k, metric)
        assert_allclose(dist1, dist2)

    for metric in DISCRETE_METRICS:
        yield check_neighbors, metric


def test_ball_tree_query_radius(n_samples=100, n_features=10):
    np.random.seed(0)
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
    np.random.seed(0)
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


def compute_kernel_slow(Y, X, kernel, h):
    d = np.sqrt(((Y[:, None, :] - X) ** 2).sum(-1))

    if kernel == 'gaussian':
        return (1. / (h * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (d * d)
                                                         / (h * h)).sum(-1)
    elif kernel == 'tophat':
        return (0.5 / h) * (d < h).sum(-1)
    elif kernel == 'epanechnikov':
        return (0.75 / h) * ((1.0 - (d * d) / (h * h)) * (d < h)).sum(-1)
    elif kernel == 'exponential':
        return (0.5 / h) * (np.exp(-d / h) * (d < h)).sum(-1)
    elif kernel == 'linear':
        return (1. / h) * ((1 - d / h) * (d < h)).sum(-1)
    elif kernel == 'cosine':
        return (0.25 * np.pi / h) * (np.cos(0.5 * np.pi * d / h)
                                     * (d < h)).sum(-1)
    else:
        raise ValueError('kernel not recognized')
    

def test_ball_tree_KDE(n_samples=100, n_features=3):
    np.random.seed(0)
    X = np.random.random((n_samples, n_features))
    Y = np.random.random((n_samples, n_features))
    bt = BallTree(X, leaf_size=10)

    for kernel in ['gaussian', 'tophat', 'epanechnikov',
                   'exponential', 'linear', 'cosine']:
        for h in [0.001, 0.01, 0.1]:
            dens_true = compute_kernel_slow(Y, X, kernel, h)
            def check_results(kernel, h, atol, rtol, dualtree, breadth_first):
                dens = bt.kernel_density(Y, h, atol=atol, rtol=rtol,
                                         kernel=kernel, dualtree=dualtree,
                                         breadth_first=breadth_first)
                assert_allclose(dens, dens_true, atol=atol, rtol=rtol)

            for rtol in [0, 1E-5]:
                for atol in [1E-10, 1E-5, 0.1]:
                    for dualtree in (True, False):
                        if dualtree and rtol > 0:
                            continue
                        for breadth_first in (True, False):
                            yield (check_results, kernel, h, atol, rtol,
                                   dualtree, breadth_first)


def test_ball_tree_two_point(n_samples=100, n_features=3):
    np.random.seed(0)
    X = np.random.random((n_samples, n_features))
    Y = np.random.random((n_samples, n_features))
    r = np.linspace(0, 1, 10)
    bt = BallTree(X, leaf_size=10)

    D = DistanceMetric.get_metric("euclidean").pairwise(Y, X)
    counts_true = [(D <= ri).sum() for ri in r]

    def check_two_point(r, dualtree):
        counts = bt.two_point_correlation(Y, r=r, dualtree=dualtree)
        assert_allclose(counts, counts_true)

    for dualtree in (True, False):
        yield check_two_point, r, dualtree


def test_ball_tree_pickle():
    import pickle
    np.random.seed(0)
    X = np.random.random((10, 3))
    bt1 = BallTree(X, leaf_size=1)
    ind1, dist1 = bt1.query(X)

    def check_pickle_protocol(protocol):
        s = pickle.dumps(bt1, protocol=protocol)
        bt2 = pickle.loads(s)
        ind2, dist2 = bt2.query(X)
        assert_allclose(ind1, ind2)
        assert_allclose(dist1, dist2)

    for protocol in (0, 1, 2):
        yield check_pickle_protocol, protocol


if __name__ == '__main__':
    import nose
    nose.runmodule()

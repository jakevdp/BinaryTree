import numpy as np
from numpy.testing import assert_allclose
from distmetrics import Distance
from scipy.spatial.distance import cdist


METRICS = {'euclidean':{},
           'cityblock':{},
           'chebyshev':{},
           'minkowski':{'p':3}}


def test_pdist():
    X = np.random.random((40, 3))
    Y = np.random.random((30, 3))

    def check_dist(metric, kwargs):
        D1 = Distance(metric, **kwargs).pairwise(X, Y)
        D2 = cdist(X, Y, metric, **kwargs)
        assert_allclose(D1, D2)

    for (metric, kwargs) in METRICS.iteritems():
        yield check_dist, metric, kwargs
        yield check_dist, metric, kwargs


if __name__ == '__main__':
    import nose
    nose.runmodule()

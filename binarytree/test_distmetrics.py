import numpy as np
from distmetrics import DistanceMetric
from scipy.spatial.distance import cdist


METRICS = {'euclidean':{},
           'cityblock':{},
           'minkowski':{'p':3}}


def check_dist(X, Y, metric, kwargs):
    D1 = DistanceMetric(metric, **kwargs).pairwise(X, Y)
    D2 = cdist(X, Y, metric, **kwargs)


def test_pdist():
    X = np.random.random((40, 3))
    Y = np.random.random((30, 3))

    for (metric, kwargs) in METRICS.iteritems():
        yield check_dist, X, Y, metric, kwargs
        yield check_dist, X, X, metric, kwargs

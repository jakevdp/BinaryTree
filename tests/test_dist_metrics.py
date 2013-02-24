import itertools

import numpy as np
from numpy.testing import assert_allclose

from scipy.spatial.distance import cdist, pdist, squareform
from ball_tree import DistanceMetric


class TestMetrics:
    metrics = {'euclidean':{},
               'cityblock':{},
               'minkowski':dict(p=(1, 1.5, 2, 3))}

    def __init__(self, n1=5, n2=6, d=4, zero_frac=0.5,
                 rseed=0, dtype=np.float64):
        np.random.seed(rseed)
        self.X1 = np.random.random((n1, d)).astype(dtype)
        self.X2 = np.random.random((n2, d)).astype(dtype)

    def test_cdist(self):
        for metric, argdict in self.metrics.iteritems():
            keys = argdict.keys()
            for vals in itertools.product(*argdict.values()):
                kwargs = dict(zip(keys, vals))
                D_true = cdist(self.X1, self.X2, metric, **kwargs)
                yield self.check_cdist, metric, kwargs, D_true
            
    def check_cdist(self, metric, kwargs, D_true):
        dm = DistanceMetric.get_metric(metric, **kwargs)
        D12 = dm.pairwise(self.X1, self.X2)
        assert_allclose(D12, D_true)

    def test_pdist(self):
        for metric, argdict in self.metrics.iteritems():
            keys = argdict.keys()
            for vals in itertools.product(*argdict.values()):
                kwargs = dict(zip(keys, vals))
                D_true = pdist(self.X1, metric, **kwargs)
                Dsq_true = squareform(D_true)
                yield self.check_pdist, metric, kwargs, Dsq_true

    def check_pdist(self, metric, kwargs, D_true):
        dm = DistanceMetric.get_metric(metric, **kwargs)
        D12 = dm.pairwise(self.X1)
        assert_allclose(D12, D_true)        

        
if __name__ == '__main__':
    import nose
    nose.runmodule()

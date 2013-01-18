from time import time

import numpy as np
from sklearn.metrics import pairwise_distances
from distmetrics import EuclideanDistance
from scipy.spatial.distance import pdist, cdist, squareform

X = np.random.random((2000, 3))

t0 = time()
D0 = cdist(X, X)
t1 = time()
print "scipy.spatial.cdist: %.3g s" % (t1 - t0)
print

t0 = time()
D1 = squareform(pdist(X))
t1 = time()
print "scipy.spatial.pdist: %.3g s" % (t1 - t0)
print "results match:", np.allclose(D0, D1)
print

t0 = time()
D2 = pairwise_distances(X)
t1 = time()
print "sklearn.metrics.pairwise: %.3g s" % (t1 - t0)
print "results match:", np.allclose(D1, D2)
print

dist = EuclideanDistance()
t0 = time()
D3 = dist.pairwise(X)
t1 = time()
print "dist.pairwise(X): %.3g s" % (t1 - t0)
print "results match:", np.allclose(D1, D3)
print

t0 = time()
D4 = dist.pairwise(X, X)
t1 = time()
print "dist.pairwise(X, X): %.3g s" % (t1 - t0)
print "results match:", np.allclose(D1, D4)
print

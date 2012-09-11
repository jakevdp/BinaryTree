import numpy as np

from ball_tree import BallTree
from kd_tree import KDTree
from sklearn.neighbors import BallTree as skBallTree
from scipy.spatial import cKDTree

from sklearn.datasets import load_digits

digits = load_digits()['data'].astype(float)
X = np.vstack([digits,
               digits[:, ::-1],
               np.hstack([digits[:, 30:], digits[:, :30]]),
               np.hstack([digits[:, 20:], digits[:, :20]])])

Xquery = np.random.random(X.shape)
print X.shape

k = 2
dualtree = False

from time import time

t0 = time()
bt = BallTree(X, 30)
dist, ind = bt.query(Xquery, k, dualtree=dualtree)
t1 = time()
print "pyDistances.BallTree: %.3g sec" % (t1 - t0)

t0 = time()
kdt = KDTree(X, 30)
dist, ind = kdt.query(Xquery, k, dualtree=dualtree)
t1 = time()
print "pyDistances.KDTree: %.3g sec" % (t1 - t0)

t0 = time()
bt = skBallTree(X, 30)
dist, ind = bt.query(Xquery, k)
t1 = time()
print "sklearn.neighbors.BallTree: %.3g sec" % (t1 - t0)

t0 = time()
kdt = cKDTree(X, 30)
dist, ind = kdt.query(Xquery, k)
t1 = time()
print "scipy.spatial.cKDTree: %.3g sec" % (t1 - t0)


from time import time
import numpy as np
from binary_tree import BallTree, KDTree

from sklearn.neighbors import BallTree as skBallTree
from scipy.spatial import cKDTree

from sklearn.datasets import fetch_olivetti_faces

k = 10
d = 5

if 0:
    faces = fetch_olivetti_faces()
    data = faces.data

    d = 5

    X = np.vstack([data[:, i * d:(i + 1) * d].astype(np.float64)
                   for i in range(5)])
    Y = np.vstack([data[:, i * d:(i + 1) * d].astype(np.float64)
                   for i in range(5, 10)])

else:
    N1 = 2000
    N2 = 2000

    X = np.random.random((N1, d)).astype(float)
    Y = np.random.random((N2, d))

print X.shape, Y.shape

bt = BallTree(X)
skbt = skBallTree(X)
ckdt = cKDTree(X)
kdt = KDTree(X)

t0 = time()
dist1, ind1 = bt.query(Y, k)
t1 = time()
dist2, ind2 = skbt.query(Y, k)
t2 = time()
dist3, ind3 = kdt.query(Y, k)
t3 = time()
dist4, ind4 = ckdt.query(Y, k)
t4 = time()

print "new: %.2g sec" % (t1 - t0)
print "old: %.2g sec" % (t2 - t1)
print "new kd: %.2g sec" % (t3 - t2)
print "old kd: %.2g sec" % (t4 - t3)
print
print "distances match:", np.all(dist1 == dist2), np.all(dist3 == dist4)
print "indices match:", np.all(ind1 == ind2), np.all(ind3 == ind4)

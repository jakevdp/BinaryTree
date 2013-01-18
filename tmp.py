from time import time
import numpy as np
from binary_tree import BallTree, _BinaryTree

from sklearn.neighbors import BallTree as skBallTree
from scipy.spatial import cKDTree

N1 = 2000
N2 = 2000
k = 10
d = 20

X = np.random.random((N1, d)).astype(float)
bt = BallTree(X)
skbt = skBallTree(X)
kdt = cKDTree(X)

Y = np.random.random((N2, d))

t0 = time()
dist1, ind1 = bt.query(Y, k)
t1 = time()
dist2, ind2 = skbt.query(Y, k)
t2 = time()
dist3, ind3 = kdt.query(Y, k)
t3 = time()

print "new: %.2g sec" % (t1 - t0)
print "old: %.2g sec" % (t2 - t1)
print "kd: %.2g sec" % (t3 - t2)
print
print "distances match:", np.all(dist1 == dist2)
print "indices match:", np.all(ind1 == ind2)

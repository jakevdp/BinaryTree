from time import time
import numpy as np
from binary_tree import BallTree, _BinaryTree
from sklearn.neighbors import BallTree as skBallTree
from ckdtree import cKDTree

from collections import defaultdict

d = 3
k = 1

times = defaultdict(lambda: [])
counts = defaultdict(lambda: [])
Nrange = 2 ** np.arange(4, 20)
N = max(Nrange)

if 0:
    from sklearn.datasets import fetch_olivetti_faces
    faces = fetch_olivetti_faces()
    data = faces.data
    num = int(np.ceil(N * 1. / data.shape[0]))
    X = np.vstack([data[:, i * d:(i + 1) * d].astype(np.float64)
                   for i in range(num)])
    Y = np.vstack([data[:, i * d:(i + 1) * d].astype(np.float64)
                   for i in range(num, num + 1)])
else:
    rseed = np.random.randint(100000)
    print("rseed = %i" % rseed)
    np.random.seed(rseed)

    X = np.random.random((N, d)).astype(float)
    Y = np.random.random((1, d)).astype(float)

Y = Y[:1]


for N in Nrange:    
    bt = BallTree(X[:N])
    skbt = skBallTree(X[:N])
    kdt = cKDTree(X[:N])

    bt.reset_dist_count()
    kdt.reset_dist_count()

    t0 = time()
    dist1, ind1 = bt.query(Y[:N], k)
    t1 = time()
    dist2, ind2 = skbt.query(Y[:N], k)
    t2 = time()
    dist3, ind3 = kdt.query(Y[:N], k)
    t3 = time()

    times['BallTree'].append(t1 - t0)
    times['sklearn BallTree'].append(t2 - t1)
    times['scipy KDTree'].append(t3 - t2)

    if not(np.all(dist1 == dist2) and np.all(dist2 == dist3)):
        print "distance do not match for N = %i" % N

    counts['BallTree'].append(bt.dist_count())
    counts['scipy KDTree'].append(kdt.dist_count())

import matplotlib.pyplot as plt

fig, ax = plt.subplots(2)

for key in times.keys():
    ax[0].loglog(Nrange, times[key], label=key)

for key in counts.keys():
    ax[1].loglog(Nrange, counts[key], label=key)

ax[0].legend(loc=2)
ax[0].grid()

ax[1].legend(loc=2)
ax[1].grid()

ax[1].set_xlabel('N')
ax[1].set_ylabel('dist calls')
ax[0].set_ylabel('t (sec)')
ax[0].set_title('d = %i, k=%i' % (d, k))

plt.show()

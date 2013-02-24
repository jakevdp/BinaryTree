from time import time
import numpy as np
import ball_tree
from ball_tree import BallTree, DTYPE, ITYPE
from sklearn.neighbors import BallTree as skBallTree


def simul_sort_numpy(dist, ind):
    i = np.argsort(dist, axis=1)
    row_ind = np.arange(dist.shape[0])[:, None]
    return dist[row_ind, i], ind[row_ind, i]
    

def bench_simultaneous_sort(n_rows=2000, n_pts=21):
    print("Simultaneous sort")
    dist1 = np.random.random((n_rows, n_pts)).astype(DTYPE)
    ind1 = (np.arange(n_pts) + np.zeros((n_rows, 1))).astype(ITYPE)

    dist2 = dist1.copy()
    ind2 = ind1.copy()
    
    t0 = time()
    dist1, ind1 = simul_sort_numpy(dist1, ind1)  # note: not an in-place sort
    t1 = time()
    ball_tree.simultaneous_sort(dist2, ind2)
    t2 = time()

    print("   numpy: %.2g sec" % (t1 - t0))
    print("   new:   %.2g sec" % (t2 - t1))
    print("   results match: (%s, %s)\n" % (np.allclose(dist1, dist2),
                                            np.allclose(ind1, ind2)))


def bench_neighbors_heap(n_rows=1000, n_pts=200, n_nbrs=21):
    print("Heap push + extracting data")
    X = np.random.random((n_rows, n_pts)).astype(DTYPE)

    t0 = time()
    I0 = np.argsort(X, 1)[:, :n_nbrs]
    D0 = X[np.arange(X.shape[0])[:, None], I0]
    t1 = time()
    D1, I1 = ball_tree.load_heap(X, n_nbrs)
    t2 = time()

    print("   memviews: %.2g sec" % (t2 - t1))
    print("   results match: (%s, %s)\n" % (np.allclose(D1, D0),
                                            np.allclose(I1, I0)))


def bench_euclidean_dist(n1=1000, n2=1100, d=3):
    print("Euclidean distances")
    X = np.random.random((n1, d)).astype(DTYPE)
    Y = np.random.random((n2, d)).astype(DTYPE)

    eucl = ball_tree.EuclideanDistance()
    
    funcs = [ball_tree.euclidean_pairwise_inline,
             ball_tree.euclidean_pairwise_class,
             ball_tree.euclidean_pairwise_polymorphic,
             eucl.pairwise]

    labels = ["inline",
              "class/direct",
              "class/polymorphic",
              "class/member func"]

    D = []
    for func, label in zip(funcs, labels):
        t0 = time()
        Di = func(X, Y)
        t1 = time()

        D.append(Di)
        print("   %s: %.2g sec" % (label, t1 - t0))

    print("   results match: (%s)\n"
          % ', '.join(['%s' % np.allclose(D[i - 1], D[i])
                       for i in range(len(D))]))


def bench_ball_tree(N=2000, D=3, k=15, leaf_size=30):
    print("Ball Tree")
    X = np.random.random((N, D)).astype(DTYPE)

    t0 = time()
    btskl = skBallTree(X, leaf_size=leaf_size)
    t1 = time()
    bt = BallTree(X, leaf_size=leaf_size)
    t2 = time()

    print("Build:")
    print("  sklearn : %.2g sec" % (t1 - t0))
    print("  new     : %.2g sec" % (t2 - t1))

    t0 = time()
    Dskl, Iskl = btskl.query(X, k)
    t1 = time()
    D1, I1 = bt.query(X, k, dualtree=False, breadth_first=False)
    t2 = time()
    D2, I2 = bt.query(X, k, dualtree=False, breadth_first=True)
    t3 = time()
    D3, I3 = bt.query(X, k, dualtree=True, breadth_first=False)
    t4 = time()
    D4, I4 = bt.query(X, k, dualtree=True, breadth_first=True)
    t5 = time()

    dist = [Dskl, D1, D2, D3, D4]
    ind  = [Iskl, I1, I2, I3, I4]

    print("Query:")
    print("  sklearn                 : %.2g sec" % (t1 - t0))
    print("  new/single/depthfirst   : %.2g sec" % (t2 - t1))
    print("  new/single/breadthfirst : %.2g sec" % (t3 - t2))
    print("  new/dual/depthfirst     : %.2g sec" % (t4 - t3))
    print("  new/dual/breadthfirst   : %.2g sec" % (t5 - t4))
    print
    print(" distances match: %s"
          % ', '.join(['%s' % np.allclose(dist[i - 1], dist[i])
                       for i in range(len(dist))]))
    print(" indices match: %s"
          % ', '.join(['%s' % np.allclose(ind[i - 1], ind[i])
                       for i in range(len(ind))]))


if __name__ == '__main__':
    bench_simultaneous_sort()
    bench_neighbors_heap()
    bench_euclidean_dist()
    bench_ball_tree()

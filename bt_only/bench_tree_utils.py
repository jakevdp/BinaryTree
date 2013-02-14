from time import time
import numpy as np
import version1.tree_utils as tree_utils_1
import version2.tree_utils as tree_utils_2
import version1.dist_metrics as dist_metrics_1
import version2.dist_metrics as dist_metrics_2

DTYPE = tree_utils_1.DTYPE
ITYPE = tree_utils_2.ITYPE


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

    dist3 = dist1.copy()
    ind3 = ind1.copy()
    
    t0 = time()
    dist1, ind1 = simul_sort_numpy(dist1, ind1)  # note: not an in-place sort
    t1 = time()
    tree_utils_1.simultaneous_sort(dist2, ind2)
    t2 = time()
    tree_utils_2.simultaneous_sort(dist3, ind3)
    t3 = time()

    print("   numpy:    %.2g sec" % (t1 - t0))
    print("   memviews: %.2g sec" % (t2 - t1))
    print("   raw ptrs: %.2g sec" % (t3 - t2))
    print("   results match: (%s, %s, %s, %s)\n" % (np.allclose(dist1, dist2),
                                                    np.allclose(ind1, ind2),
                                                    np.allclose(dist1, dist3),
                                                    np.allclose(ind1, ind3)))


def bench_neighbors_heap(n_rows=1000, n_pts=200, n_nbrs=21):
    print("Heap push + extracting data")
    X = np.random.random((n_rows, n_pts)).astype(DTYPE)

    t0 = time()
    I0 = np.argsort(X, 1)[:, :n_nbrs]
    D0 = X[np.arange(X.shape[0])[:, None], I0]
    t1 = time()
    D1, I1 = tree_utils_1.load_heap(X, n_nbrs)
    t2 = time()
    D2, I2 = tree_utils_2.load_heap(X, n_nbrs)
    t3 = time()

    print("   memviews: %.2g sec" % (t2 - t1))
    print("   raw ptrs: %.2g sec" % (t3 - t2))
    print("   results match: (%s, %s, %s, %s)\n" % (np.allclose(D1, D0),
                                                    np.allclose(I1, I0),
                                                    np.allclose(D2, D0),
                                                    np.allclose(I2, I0)))


def bench_euclidean_dist(n1=1000, n2=1100, d=3):
    print("Euclidean distances")
    X = np.random.random((n1, d)).astype(DTYPE)
    Y = np.random.random((n2, d)).astype(DTYPE)

    eucl_1 = dist_metrics_1.EuclideanDistance()
    eucl_2 = dist_metrics_2.EuclideanDistance()
    
    t = []
    D = []
    for func in [dist_metrics_1.euclidean_pairwise,
                 dist_metrics_1.euclidean_pairwise_class,
                 dist_metrics_1.euclidean_pairwise_polymorph,
                 eucl_1.pairwise,
                 dist_metrics_2.euclidean_pairwise,
                 dist_metrics_2.euclidean_pairwise_class,
                 dist_metrics_2.euclidean_pairwise_polymorph,
                 eucl_2.pairwise]:
        t0 = time()
        Di = func(X, Y)
        t1 = time()

        D.append(Di)
        t.append(t1 - t0)

    print("   memview/inline: %.2g sec" % t[0])
    print("   memview/class/inline/direct: %.2g sec" % t[1])
    print("   memview/class/inline/polymorph: %.2g sec" % t[2])
    print("   memview/class/not inline:  %.2g sec" % t[3])
    print('')
    print("   pointers/inline: %.2g sec" % t[4])
    print("   pointers/class/inline/direct: %.2g sec" % t[5])
    print("   pointers/class/inline/polymorph: %.2g sec" % t[6])
    print("   pointers/class/not inline:  %.2g sec" % t[7])
    print("   results match: (%s)\n" % ', '.join(
            ['%s' % np.allclose(D[i], D[i + 1])
             for i in range(len(D) - 1)]))


if __name__ == '__main__':
    bench_simultaneous_sort()
    bench_neighbors_heap()
    bench_euclidean_dist()

from time import time
import numpy as np
from sklearn.neighbors import BallTree as skBallTree
from ball_tree import BallTree
from kd_tree import KDTree

def time_func(executable, *args, **kwargs):
    N = 5
    t = 0
    for i in range(N):
        t0 = time()
        executable(*args, **kwargs)
        t1 = time()
        t += (t1 - t0)
    return t * 1. / N


def bench_trees(N=2000, D=3, leaf_size=40):
    X = np.random.random((N, D))

    labels = ['skBallTree', 'BallTree', 'KDTree']
    tree_types = [skBallTree, BallTree, KDTree]

    print '%s %s %s %s %s' % tuple(map(lambda s: s.ljust(15),
                                       ('', 'Construction',
                                        'KNN', 'RNN', 'KDE')))
    for tree_type, label in zip(tree_types, labels):
        tree = tree_type(X, leaf_size=leaf_size)

        construct_time = time_func(tree_type, X, leaf_size=leaf_size)
        knn_time = time_func(tree.query, X, k=5)
        rnn_time = time_func(tree.query_radius, X, r=0.1)

        if not label.startswith('sk'):
            kde_time = time_func(tree.kernel_density, X,
                                 h=0.01, rtol=0.001, atol=0.001)
            print '%s %s %s %s %s' % tuple(map(lambda s: s.ljust(15),
                                               (label,
                                                '%.2g' % construct_time,
                                                '%.2g' % knn_time,
                                                '%.2g' % rnn_time,
                                                '%.2g' % kde_time)))
        else:
            print '%s %s %s %s' % tuple(map(lambda s: s.ljust(15),
                                            (label,
                                             '%.2g' % construct_time,
                                             '%.2g' % knn_time,
                                             '%.2g' % rnn_time)))

if __name__ == '__main__':
    bench_trees()

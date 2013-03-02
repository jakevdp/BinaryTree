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


def bench_trees(N=2000, D=3, leaf_size=40, metric='euclidean', **kwargs):
    print "{N} points, metric = {metric}".format(N=N, metric=metric)

    X = np.random.random((N, D))

    labels = ['skBallTree', 'BallTree', 'KDTree']
    tree_types = [skBallTree, BallTree, KDTree]

    spacing = 11

    print '%s %s %s %s %s %s' % tuple(map(lambda s: s.ljust(spacing),
                                          ('', 'Build',
                                           'KNN', 'RNN', 'KDE', '2PT')))
    for tree_type, label in zip(tree_types, labels):
        if label == 'skBallTree':
            tree = tree_type(X, leaf_size=leaf_size)
            construct_time = time_func(tree_type, X, leaf_size=leaf_size)
        else:
            tree = tree_type(X, leaf_size=leaf_size,
                             metric=metric, **kwargs)
            construct_time = time_func(tree_type, X, leaf_size=leaf_size,
                                       metric=metric, **kwargs)
        knn_time = time_func(tree.query, X, k=3)
        rnn_time = time_func(tree.query_radius, X, r=0.1)

        if not label.startswith('sk'):
            kde_time = time_func(tree.kernel_density, X,
                                 h=0.01, rtol=0.001, atol=0.001)
            twopt_time = time_func(tree.two_point_correlation, X,
                                   r = np.linspace(0, 1, 100))
            print '%s %s %s %s %s %s' % tuple(map(lambda s: s.ljust(spacing),
                                                  (label,
                                                   '%.2g' % construct_time,
                                                   '%.2g' % knn_time,
                                                   '%.2g' % rnn_time,
                                                   '%.2g' % kde_time,
                                                   '%.2g' % twopt_time)))
        else:
            print '%s %s %s %s' % tuple(map(lambda s: s.ljust(spacing),
                                            (label,
                                             '%.2g' % construct_time,
                                             '%.2g' % knn_time,
                                             '%.2g' % rnn_time)))
    print


if __name__ == '__main__':
    bench_trees(1000, metric='euclidean')
    bench_trees(1000, metric='minkowski', p=3)

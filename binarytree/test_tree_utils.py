import numpy as np
from numpy.testing import assert_allclose
from tree_utils import MaxHeap, sort_dist_idx, DTYPE, ITYPE

def test_max_heap(N=50, k=10):
    m = MaxHeap(k)

    d_in = np.random.random(N).astype(DTYPE)
    i_in = np.arange(N, dtype=ITYPE)
    for d, i in zip(d_in, i_in):
        m.push(d, i)

    ind = np.argsort(d_in)
    d_in = d_in[ind]
    i_in = i_in[ind]

    d_heap, i_heap = m.get_arrays(sort=True)

    assert_allclose(d_in[:k], d_heap)
    assert_allclose(i_in[:k], i_heap)


def test_max_heap_wrap(N=50, k=10):
    # wrap existing arrays
    m = MaxHeap()
    d_heap = np.zeros(k, dtype=DTYPE) + np.inf
    i_heap = np.zeros(k, dtype=ITYPE)
    m.wrap(d_heap, i_heap)

    d_in = np.random.random(N).astype(DTYPE)
    i_in = np.arange(N, dtype=ITYPE)
    for d, i in zip(d_in, i_in):
        m.push(d, i)

    ind = np.argsort(d_in)
    d_in = d_in[ind]
    i_in = i_in[ind]

    # in-place sort: d_heap and i_heap should end up sorted
    tmp1, tmp2 = m.get_arrays(sort=True)

    assert_allclose(d_in[:k], d_heap)
    assert_allclose(i_in[:k], i_heap)


def test_sort_dist_idx(N=201):
    dist = np.random.random(N).astype(DTYPE)
    ind = np.arange(N, dtype=ITYPE)

    dist2 = dist.copy()
    ind2 = ind.copy()

    sort_dist_idx(dist, ind)
    
    i = np.argsort(dist2)
    dist2 = dist2[i]
    ind2 = ind2[i]

    assert_allclose(dist, dist2)
    assert_allclose(ind, ind2)

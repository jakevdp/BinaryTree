import numpy as np
from numpy.testing import assert_allclose, assert_
from tree_utils import MaxHeap, sort_dist_idx, find_split_dim,\
    partition_indices, DTYPE, ITYPE

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


def test_find_split_dim():
    # check for several different random seeds
    for i in range(5):
        np.random.seed(i)
        data = np.random.random((50, 5)).astype(DTYPE)
        indices = np.arange(50, dtype=ITYPE)
        np.random.shuffle(indices)

        idx_start = 10
        idx_end = 40

        i_split = find_split_dim(data, indices, idx_start, idx_end)
        i_split_2 = np.argmax(np.max(data[indices[idx_start:idx_end]], 0) -
                              np.min(data[indices[idx_start:idx_end]], 0))

        assert_(i_split == i_split_2)


def test_partition_indices():
    # check for several different random seeds
    for i in range(5):
        np.random.seed(i)

        data = np.random.random((50, 5)).astype(DTYPE)
        indices = np.arange(50, dtype=ITYPE)
        np.random.shuffle(indices)

        split_dim = 2
        split_index = 25
        idx_start = 10
        idx_end = 40

        partition_indices(data, indices, split_dim,
                          split_index, idx_start, idx_end)
    
        assert_(np.all(data[indices[idx_start:split_index], split_dim]
                       <= data[indices[split_index], split_dim]) and
                np.all(data[indices[split_index], split_dim]
                       <= data[indices[split_index:idx_end], split_dim]))
    


if __name__ == '__main__':
    import nose
    nose.runmodule()

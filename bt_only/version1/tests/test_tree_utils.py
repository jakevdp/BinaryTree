import numpy as np
from numpy.testing import assert_allclose, assert_
from version1.tree_utils import NeighborsHeap, simultaneous_sort, DTYPE, ITYPE

def test_neighbors_heap(n_pts=5, n_nbrs=10):
    heap = NeighborsHeap(n_pts, n_nbrs)

    for row in range(n_pts):
        d_in = np.random.random(2 * n_nbrs).astype(DTYPE)
        i_in = np.arange(2 * n_nbrs, dtype=ITYPE)
        for d, i in zip(d_in, i_in):
            heap.push(row, d, i)

        ind = np.argsort(d_in)
        d_in = d_in[ind]
        i_in = i_in[ind]

        d_heap, i_heap = heap.get_arrays(sort=True)

        assert_allclose(d_in[:n_nbrs], d_heap[row])
        assert_allclose(i_in[:n_nbrs], i_heap[row])


def test_simultaneous_sort(n_rows=10, n_pts=201):
    dist = np.random.random((n_rows, n_pts)).astype(DTYPE)
    ind = (np.arange(n_pts) + np.zeros((n_rows, 1))).astype(ITYPE)

    dist2 = dist.copy()
    ind2 = ind.copy()

    # simultaneous sort rows using function
    simultaneous_sort(dist, ind)
    
    # simultaneous sort rows using numpy
    i = np.argsort(dist2, axis=1)
    row_ind = np.arange(n_rows)[:, None]
    dist2 = dist2[row_ind, i]
    ind2 = ind2[row_ind, i]

    assert_allclose(dist, dist2)
    assert_allclose(ind, ind2)


def _test_find_split_dim():
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


def _test_partition_indices():
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
                          idx_start, split_index, idx_end)
    
        assert_(np.all(data[indices[idx_start:split_index], split_dim]
                       <= data[indices[split_index], split_dim]) and
                np.all(data[indices[split_index], split_dim]
                       <= data[indices[split_index:idx_end], split_dim]))
    


if __name__ == '__main__':
    import nose
    nose.runmodule()

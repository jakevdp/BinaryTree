# Max-heap for keeping track of neighbors
#
#  This is a basic implementation of a fixed-size binary max-heap.
#  It can be used in place of priority_queue to keep track of the
#  k-nearest neighbors in a query.  The implementation is faster than
#  priority_queue for a very large number of neighbors (k > 50 or so).
#  The implementation is slower than priority_queue for fewer neighbors.
#  The other disadvantage is that for max_heap, the indices/distances must
#  be sorted upon completion of the query.  In priority_queue, the indices
#  and distances are sorted without an extra call.
#
#  The root node is at heap[0].  The two child nodes of node i are at
#  (2 * i + 1) and (2 * i + 2).
#  The parent node of node i is node floor((i-1)/2).  Node 0 has no parent.
#  A max heap has (heap[i] >= heap[2 * i + 1]) and (heap[i] >= heap[2 * i + 2])
#  for all valid indices.
#
#  In this implementation, an empty heap should be full of infinities
#
cdef struct HeapData:
    DTYPE_t* val
    ITYPE_t* idx
    ITYPE_t size


cdef inline void heap_init(HeapData* heapdata, DTYPE_t* val,
                           ITYPE_t* idx, ITYPE_t size):
    heapdata.val = val
    heapdata.idx = idx
    heapdata.size = size


cdef inline int heap_needs_final_sort(HeapData* heapdata):
    return 1


cdef inline DTYPE_t heap_largest(HeapData* heapdata):
    return heapdata.val[0]


cdef inline ITYPE_t heap_idx_largest(HeapData* heapdata):
    return heapdata.idx[0]


cdef inline void heap_insert(HeapData* heapdata, DTYPE_t val, ITYPE_t i_val):
    cdef ITYPE_t i, ic1, ic2, i_tmp
    cdef DTYPE_t d_tmp

    # check if val should be in heap
    if val > heapdata.val[0]:
        return

    # insert val at position zero
    heapdata.val[0] = val
    heapdata.idx[0] = i_val

    #descend the heap, swapping values until the max heap criterion is met
    i = 0
    while 1:
        ic1 = 2 * i + 1
        ic2 = ic1 + 1

        if ic1 >= heapdata.size:
            break
        elif ic2 >= heapdata.size:
            if heapdata.val[ic1] > val:
                i_swap = ic1
            else:
                break
        elif heapdata.val[ic1] >= heapdata.val[ic2]:
            if val < heapdata.val[ic1]:
                i_swap = ic1
            else:
                break
        else:
            if val < heapdata.val[ic2]:
                i_swap = ic2
            else:
                break

        heapdata.val[i] = heapdata.val[i_swap]
        heapdata.idx[i] = heapdata.idx[i_swap]

        i = i_swap

    heapdata.val[i] = val
    heapdata.idx[i] = i_val

# Priority Queue for keeping track of neighbors
#
#  This is used to keep track of the neighbors as they are found.
#  It keeps the list of neighbors sorted, and inserts each new item
#  into the list.  In this fixed-size implementation, empty elements
#  are represented by infinities.
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
    return 0


cdef inline DTYPE_t heap_largest(HeapData* heapdata):
    return heapdata.val[heapdata.size - 1]


cdef inline ITYPE_t heap_idx_largest(HeapData* heapdata):
    return heapdata.idx[heapdata.size - 1]


cdef inline void heap_insert(HeapData* heapdata, DTYPE_t val, ITYPE_t i_val):
    cdef ITYPE_t i_lower = 0
    cdef ITYPE_t i_upper = heapdata.size - 1
    cdef ITYPE_t i, i_mid

    if val >= heapdata.val[i_upper]:
        return
    elif val <= heapdata.val[i_lower]:
        i_mid = i_lower
    else:
        while True:
            if (i_upper - i_lower) < 2:
                i_mid = i_lower + 1
                break
            else:
                i_mid = (i_lower + i_upper) / 2

            if i_mid == i_lower:
                i_mid += 1
                break

            if val >= heapdata.val[i_mid]:
                i_lower = i_mid
            else:
                i_upper = i_mid

    for i from heapdata.size > i > i_mid:
        heapdata.val[i] = heapdata.val[i - 1]
        heapdata.idx[i] = heapdata.idx[i - 1]

    heapdata.val[i_mid] = val
    heapdata.idx[i_mid] = i_val

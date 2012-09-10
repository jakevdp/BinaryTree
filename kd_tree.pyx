#----------------------------------------------------------------------
# Module-level documentation
include "README.pxi"

#----------------------------------------------------------------------
# Common imports and definitions
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport fmax, fmin, fabs

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

# warning: there will be problems if ITYPE
#  is switched to an unsigned type!
ITYPE = np.intp
ctypedef np.intp_t ITYPE_t

#----------------------------------------------------------------------
# Distance function definition
include "euclidean_distance.pxi"

#----------------------------------------------------------------------
# Type of tree to construct
include "kd_tree_defs.pxi"

#----------------------------------------------------------------------
# Type of heap to use
#include "priority_queue.pxi"
include "max_heap.pxi"

#----------------------------------------------------------------------
# BinaryTree base class
include "binary_tree.pxi"

#----------------------------------------------------------------------
# Create the BallTree class: this inherits from _BinaryTree
cdef class KDTree(_BinaryTree):
    __doc__ = """KD Tree Documentation"""
    def __init__(self, X, leaf_size=20):
        self.__init_common(X, leaf_size)

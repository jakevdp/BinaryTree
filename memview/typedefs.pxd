cimport numpy as np

# Floating point/data type
ctypedef np.float64_t DTYPE_t

# Index/integer type.
#  warning: there will be problems if ITYPE is switched to an unsigned type!
ctypedef np.intp_t ITYPE_t

# Fused type for certain operations
ctypedef fused DITYPE_t:
    ITYPE_t
    DTYPE_t

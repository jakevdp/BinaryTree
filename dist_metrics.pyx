#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as np
cimport numpy as np

# some handy constants
from libc.math cimport fmax, fmin, fabs, sqrt, exp, cos, pow
cdef DTYPE_t INF = np.inf

from typedefs cimport DTYPE_t, ITYPE_t, DITYPE_t
from typedefs import DTYPE, ITYPE


######################################################################
# newObj function
#  this is a helper function for pickling
def newObj(obj):
    return obj.__new__(obj)


######################################################################
# Distance Metric Classes
cdef class DistanceMetric:
    """DistanceMetric class

    This class provides a uniform interface to fast distance metric
    functions.  The various metrics can be accessed via the `get_metric`
    class method and the metric string identifier (see below).
    For example, to use the Euclidean distance:

    >>> dist = DistanceMetric.get_metric('euclidean')
    >>> X = [[0, 1, 2],
             [3, 4, 5]])
    >>> dist.pairwise(X)
    array([[ 0.        ,  5.19615242],
           [ 5.19615242,  0.        ]])

    Available Metrics
    -----------------
    The following lists the string metric identifiers and the associated
    distance metric classes:

    *Metrics intended for real-valued vector spaces:*

    ==============  ====================  ========  ===========================
    identifier      class name            args      distance function
    --------------  --------------------  --------  ---------------------------
    "euclidean"     EuclideanDistance     -         sqrt(sum((x - y)^2))
    "manhattan"     ManhattanDistance     -         sum(|x - y|)
    "chebyshev"     ChebyshevDistance     -         sum(max(|x - y|))
    "minkowski"     MinkowskiDistance     p         sum(|x - y|^p)^(1/p)
    "wminkowski"    WMinkowskiDistance    p, w      sum(w * |x - y|^p)^(1/p)
    "seuclidean"    SEuclideanDistance    V         sqrt(sum((x - y)^2 / V))
    "mahalanobis"   MahalanobisDistance   V or VI   sqrt((x - y)' V^-1 (x - y))
    ==============  ====================  ========  ===========================

    *Metrics intended for integer-valued vector spaces:*  Though intended
    for integer-valued vectors, these are also valid metrics in the case of
    real-valued vectors.

    =============  ====================  ====================================
    identifier     class name            distance function
    -------------  --------------------  ------------------------------------
    "hamming"      HammingDistance       N_unequal(x, y) / N_tot
    "canberra"     CanberraDistance      sum(|x - y| / (|x| + |y|))
    "braycurtis"   BrayCurtisDistance    sum(|x - y|) / (sum(|x|) + sum(|y|))
    =============  ====================  ====================================

    *Metrics intended for boolean-valued vector spaces:*  Any nonzero entry
    is evaluated to "True".  In the listings below, the following
    abbreviations are used:

     - N  : number of dimensions
     - NTT : number of dims in which both values are True
     - NTF : number of dims in which the first value is True, second is False
     - NFT : number of dims in which the first value is False, second is True
     - NFF : number of dims in which both values are False
     - NNEQ : number of non-equal dimensions, NNEQ = NTF + NFT
     - NNZ : number of nonzero dimensions, NNZ = NTF + NFT + NTT

    =============     ======================  =================================
    identifier        class name              distance function
    -------------     ----------------------  ---------------------------------
    "jaccard"         JaccardDistance         NNEQ / NNZ
    "maching"         MatchingDistance        NNEQ / N
    "dice"            DiceDistance            NNEQ / (NTT + NNZ)
    "kulsinski"       KulsinskiDistance       (NNEQ + N - NTT) / (NNEQ + N)
    "rogerstanimoto"  RogerStanimotoDistance  2 * NNEQ / (N + NNEQ)
    "russellrao"      RussellRaoDistance      NNZ / N
    "sokalmichener"   SokalMichenerDistance   2 * NNEQ / (N + NNEQ)
    "sokalsneath"     SokalSneathDistance     NNEQ / (NNEQ + 0.5 * NTT)
    ================  ======================  =================================
    """
    def __cinit__(self):
        self.p = 2
        self.vec = np.zeros(1, dtype=DTYPE)
        self.mat = np.zeros((1, 1), dtype=DTYPE)
        self.vec_ptr = &self.vec[0]
        self.mat_ptr = &self.mat[0, 0]
        self.size = 1

    def __reduce__(self):
        """
        reduce method used for pickling
        """
        return (newObj, (self.__class__,), self.__getstate__())

    def __getstate__(self):
        """
        get state for pickling
        """
        return (float(self.p),
                np.asarray(self.vec),
                np.asarray(self.mat))

    def __setstate__(self, state):
        """
        set state for pickling
        """
        self.p = state[0]
        self.vec = state[1]
        self.mat = state[2]
        self.vec_ptr = &self.vec[0]
        self.mat_ptr = &self.mat[0, 0]
        self.size = 1

    @classmethod
    def get_metric(cls, metric, **kwargs):
        """Get the given distance metric from the string identifier.

        See the docstring of DistanceMetric for a list of available metrics.

        Parameters
        ----------
        metric : string or class name
            The distance metric to use
        **kwargs
            additional arguments will be passed to the requested metric
        """
        if isinstance(metric, DistanceMetric):
            return metric
        elif isinstance(metric, type) and issubclass(metric, DistanceMetric):
            return metric(**kwargs)
        elif metric in [None, 'euclidean', 'l2']:
            return EuclideanDistance()
        elif metric in ['minkowski', 'p']:
            p = kwargs.get('p', 2)
            if p == 1:
                return ManhattanDistance()
            if p == 2:
                return EuclideanDistance()
            elif np.isinf(p):
                return ChebyshevDistance()
            else:
                return MinkowskiDistance(p)
        elif metric in ['manhattan', 'cityblock', 'l1']:
            return ManhattanDistance()
        elif metric in ['chebyshev', 'infinity']:
            return ChebyshevDistance()
        elif metric == 'seuclidean':
            return SEuclideanDistance(**kwargs)
        elif metric == 'mahalanobis':
            return MahalanobisDistance(**kwargs)
        elif metric == 'wminkowski':
            return WMinkowskiDistance(**kwargs)
        elif metric == 'hamming':
            return HammingDistance(**kwargs)
        elif metric == 'canberra':
            return CanberraDistance(**kwargs)
        elif metric == 'braycurtis':
            return BrayCurtisDistance(**kwargs)
        elif metric == 'matching':
            return MatchingDistance()
        elif metric == 'hamming':
            return HammingDistance()
        elif metric == 'jaccard':
            return JaccardDistance()
        elif metric == 'dice':
            return DiceDistance()
        elif metric == 'kulsinski':
            return KulsinskiDistance()
        elif metric == 'rogerstanimoto':
            return RogerStanimotoDistance()
        elif metric == 'russellrao':
            return RussellRaoDistance()
        elif metric == 'sokalmichener':
            return SokalMichenerDistance()
        elif metric == 'sokalsneath':
            return SokalSneathDistance()
        else:
            raise ValueError('metric = "%s" not recognized' % str(metric))

    def __init__(self, **kwargs):
        if self.__class__ is DistanceMetric:
            raise NotImplementedError("DistanceMetric is an abstract class")

    cdef DTYPE_t dist(self, DTYPE_t* x1, DTYPE_t* x2, ITYPE_t size):
        return -999

    cdef DTYPE_t rdist(self, DTYPE_t* x1, DTYPE_t* x2, ITYPE_t size):
        return self.dist(x1, x2, size)

    cdef DTYPE_t[:, ::1] pdist(self, DTYPE_t[:, ::1] X):
        cdef ITYPE_t i1, i2
        cdef DTYPE_t[:, ::1] D = np.zeros((X.shape[0], X.shape[0]),
                                          dtype=DTYPE, order='C')
        for i1 in range(X.shape[0]):
            for i2 in range(i1, X.shape[0]):
                D[i1, i2] = self.dist(&X[i1, 0], &X[i2, 0], X.shape[1])
                D[i2, i1] = D[i1, i2]
        return D

    cdef DTYPE_t[:, ::1] cdist(self, DTYPE_t[:, ::1] X, DTYPE_t[:, ::1] Y):
        cdef ITYPE_t i1, i2
        if X.shape[1] != Y.shape[1]:
            raise ValueError('X and Y must have the same second dimension')

        cdef DTYPE_t[:, ::1] D = np.zeros((X.shape[0], Y.shape[0]),
                                          dtype=DTYPE, order='C')

        for i1 in range(X.shape[0]):
            for i2 in range(Y.shape[0]):
                D[i1, i2] = self.dist(&X[i1, 0], &Y[i2, 0], X.shape[1])
        return D

    cdef DTYPE_t rdist_to_dist(self, DTYPE_t rdist):
        return rdist

    cdef DTYPE_t dist_to_rdist(self, DTYPE_t dist):
        return dist

    def rdist_to_dist_arr(self, rdist):
        return rdist

    def dist_to_rdist_arr(self, dist):
        return dist

    def pairwise(self, X, Y=None):
        X = np.asarray(X, dtype=DTYPE)
        if Y is None:
            D = self.pdist(X)
        else:
            Y = np.asarray(Y, dtype=DTYPE)
            D = self.cdist(X, Y)
        return np.asarray(D)


#------------------------------------------------------------
# Euclidean Distance
#  d = sqrt(sum(x_i^2 - y_i^2))
cdef class EuclideanDistance(DistanceMetric):
    """Euclidean Distance metric

    .. math::
       D(x, y) = \sqrt{ \sum_i (x_i - y_i) ^ 2 }
    """
    def __init__(self):
        self.p = 2

    cdef inline DTYPE_t dist(self, DTYPE_t* x1, DTYPE_t* x2, ITYPE_t size):
        return euclidean_dist(x1, x2, size)

    cdef inline DTYPE_t rdist(self, DTYPE_t* x1, DTYPE_t* x2, ITYPE_t size):
        return euclidean_rdist(x1, x2, size)

    cdef inline DTYPE_t rdist_to_dist(self, DTYPE_t rdist):
        return sqrt(rdist)

    cdef inline DTYPE_t dist_to_rdist(self, DTYPE_t dist):
        return dist * dist

    def rdist_to_dist_arr(self, rdist):
        return np.sqrt(rdist)

    def dist_to_rdist_arr(self, dist):
        return dist ** 2


#------------------------------------------------------------
# SEuclidean Distance
#  d = sqrt(sum((x_i - y_i2)^2 / v_i))
cdef class SEuclideanDistance(DistanceMetric):
    """Standardized Euclidean Distance metric

    .. math::
       D(x, y) = \sqrt{ \sum_i \frac{ (x_i - y_i) ^ 2}{V_i} }
    """
    def __init__(self, V):
        self.vec = np.asarray(V, dtype=DTYPE)
        self.size = self.vec.shape[0]
        self.vec_ptr = &self.vec[0]
        self.p = 2

    cdef inline DTYPE_t rdist(self, DTYPE_t* x1, DTYPE_t* x2, ITYPE_t size):
        if size != self.size:
            raise ValueError('SEuclidean dist: size of V does not match')
        cdef DTYPE_t tmp, d=0
        for j in range(size):
            tmp = x1[j] - x2[j]
            d += tmp * tmp / self.vec_ptr[j]
        return d

    cdef inline DTYPE_t dist(self, DTYPE_t* x1, DTYPE_t* x2, ITYPE_t size):
        return sqrt(self.rdist(x1, x2, size))

    cdef inline DTYPE_t rdist_to_dist(self, DTYPE_t rdist):
        return sqrt(rdist)

    cdef inline DTYPE_t dist_to_rdist(self, DTYPE_t dist):
        return dist * dist

    def rdist_to_dist_arr(self, rdist):
        return np.sqrt(rdist)

    def dist_to_rdist_arr(self, dist):
        return dist ** 2


#------------------------------------------------------------
# Manhattan Distance
#  d = sum(abs(x_i - y_i))
cdef class ManhattanDistance(DistanceMetric):
    """Manhattan/City-block Distance metric

    .. math::
       D(x, y) = \sum_i |x_i - y_i|
    """
    def __init__(self):
        self.p = 1

    cdef inline DTYPE_t dist(self, DTYPE_t* x1, DTYPE_t* x2, ITYPE_t size):
        cdef DTYPE_t d = 0
        for j in range(size):
            d += fabs(x1[j] - x2[j])
        return d


#------------------------------------------------------------
# Chebyshev Distance
#  d = max_i(abs(x_i), abs(y_i))
cdef class ChebyshevDistance(DistanceMetric):
    """Chebyshev/Infinity Distance

    .. math::
       D(x, y) = max_i (|x_i - y_i|)
    """
    def __init__(self):
        self.p = INF

    cdef inline DTYPE_t dist(self, DTYPE_t* x1, DTYPE_t* x2, ITYPE_t size):
        cdef DTYPE_t d = 0
        for j in range(size):
            d = fmax(d, fabs(x1[j] - x2[j]))
        return d


#------------------------------------------------------------
# Minkowski Distance
#  d = sum(x_i^p - y_i^p) ^ (1/p)
cdef class MinkowskiDistance(DistanceMetric):
    """Minkowski Distance

    .. math::
       D(x, y) = [\sum_i (x_i - y_i)^p] ^ (1/p)

    Minkowski Distance requires p >= 1 and finite. For p = infinity,
    use ChebyshevDistance.
    Note that for p=1, ManhattanDistance is more efficient, and for
    p=2, EuclideanDistance is more efficient.
    """
    def __init__(self, p):
        if p < 1:
            raise ValueError("p must be greater than 1")
        elif np.isinf(p):
            raise ValueError("MinkowskiDistance requires finite p. "
                             "For p=inf, use ChebyshevDistance.")
        self.p = p

    cdef inline DTYPE_t rdist(self, DTYPE_t* x1, DTYPE_t* x2, ITYPE_t size):
        cdef DTYPE_t d=0
        for j in range(size):
            d += pow(fabs(x1[j] - x2[j]), self.p)
        return d

    cdef inline DTYPE_t dist(self, DTYPE_t* x1, DTYPE_t* x2, ITYPE_t size):
        return pow(self.rdist(x1, x2, size), 1. / self.p)

    cdef inline DTYPE_t rdist_to_dist(self, DTYPE_t rdist):
        return pow(rdist, 1. / self.p)

    cdef inline DTYPE_t dist_to_rdist(self, DTYPE_t dist):
        return pow(dist, self.p)

    def rdist_to_dist_arr(self, rdist):
        return rdist ** (1. / self.p)

    def dist_to_rdist_arr(self, dist):
        return dist ** self.p


#------------------------------------------------------------
# W-Minkowski Distance
#  d = sum(w_i * (x_i^p - y_i^p)) ^ (1/p)
cdef class WMinkowskiDistance(DistanceMetric):
    """Weighted Minkowski Distance

    .. math::
       D(x, y) = [\sum_i (x_i - y_i)^p] ^ (1/p)

    Weighted Minkowski Distance requires p >= 1 and finite. 
    """
    def __init__(self, p, w):
        if p < 1:
            raise ValueError("p must be greater than 1")
        elif np.isinf(p):
            raise ValueError("MinkowskiDistance requires finite p. "
                             "For p=inf, use ChebyshevDistance.")
        self.p = p
        self.vec = np.asarray(w, dtype=DTYPE)
        self.size = self.vec.shape[0]
        self.vec_ptr = &self.vec[0]

    cdef inline DTYPE_t rdist(self, DTYPE_t* x1, DTYPE_t* x2, ITYPE_t size):
        if size != self.size:
            raise ValueError('SEuclidean dist: size of V does not match')
        cdef DTYPE_t d=0
        for j in range(size):
            d += pow(self.vec_ptr[j] * fabs(x1[j] - x2[j]), self.p)
        return d

    cdef inline DTYPE_t dist(self, DTYPE_t* x1, DTYPE_t* x2, ITYPE_t size):
        return pow(self.rdist(x1, x2, size), 1. / self.p)

    cdef inline DTYPE_t rdist_to_dist(self, DTYPE_t rdist):
        return pow(rdist, 1. / self.p)

    cdef inline DTYPE_t dist_to_rdist(self, DTYPE_t dist):
        return pow(dist, self.p)

    def rdist_to_dist_arr(self, rdist):
        return rdist ** (1. / self.p)

    def dist_to_rdist_arr(self, dist):
        return dist ** self.p


#------------------------------------------------------------
# Mahalanobis Distance
#  d = sqrt( (x - y)^T V^-1 (x - y) )
cdef class MahalanobisDistance(DistanceMetric):
    """Mahalanobis Distance

    .. math::
       D(x, y) = \sqrt{ (x - y)^T V^{-1} (x - y) }

    Parameters
    ----------
    V : array_like
        Symmetric positive-definite covariance matrix.
        The inverse of this matrix will be explicitly computed.
    VI : array_like
        optionally specify the inverse directly.  If VI is passed,
        then V is not referenced.
    """
    def __init__(self, V=None, VI=None):
        if VI is None:
            VI = np.linalg.inv(V)
        if VI.ndim != 2 or VI.shape[0] != VI.shape[1]:
            raise ValueError("V/VI must be square")
        self.mat = np.asarray(VI, dtype=float, order='C')
        self.mat_ptr = &self.mat[0, 0]
        self.size = self.mat.shape[0]

        # we need vec as a work buffer
        self.vec = np.zeros(self.size, dtype=DTYPE)
        self.vec_ptr = &self.vec[0]

    cdef inline DTYPE_t rdist(self, DTYPE_t* x1, DTYPE_t* x2, ITYPE_t size):
        if size != self.size:
            raise ValueError('Mahalanobis dist: size of V does not match')

        cdef DTYPE_t tmp, d = 0
        
        # compute (x1 - x2).T * VI * (x1 - x2)
        for i in range(size):
            self.vec_ptr[i] = x1[i] - x2[i]
        
        for i in range(size):
            tmp = 0
            for j in range(size):
                tmp += self.mat_ptr[i * size + j] * self.vec_ptr[j]
            d += tmp * self.vec_ptr[i]
        return d

    cdef inline DTYPE_t dist(self, DTYPE_t* x1, DTYPE_t* x2, ITYPE_t size):
        return sqrt(self.rdist(x1, x2, size))

    cdef inline DTYPE_t rdist_to_dist(self, DTYPE_t rdist):
        return sqrt(rdist)

    cdef inline DTYPE_t dist_to_rdist(self, DTYPE_t dist):
        return dist * dist

    def rdist_to_dist_arr(self, rdist):
        return np.sqrt(rdist)

    def dist_to_rdist_arr(self, dist):
        return dist ** 2


#------------------------------------------------------------
# Hamming Distance
#  d = N_unequal(x, y) / N_tot
cdef class HammingDistance(DistanceMetric):
    """Hamming Distance

    Hamming distance is meant for discrete-valued vectors, though it is
    a valid metric for real-valued vectors.

    .. math::
       D(x, y) = \frac{1}{N} \sum_i \delta_{x_i, y_i}
    """
    cdef inline DTYPE_t dist(self, DTYPE_t* x1, DTYPE_t* x2, ITYPE_t size):
        cdef int n_unequal = 0
        for j in range(size):
            if x1[j] != x2[j]:
                n_unequal += 1
        return float(n_unequal) / size


#------------------------------------------------------------
# Canberra Distance
#  D(x, y) = sum[ abs(x_i - y_i) / (abs(x_i) + abs(y_i)) ]
cdef class CanberraDistance(DistanceMetric):
    """Canberra Distance

    Canberra distance is meant for discrete-valued vectors, though it is
    a valid metric for real-valued vectors.

    .. math::
       D(x, y) = \sum_i \frac{|x_i - y_i|}{|x_i| + |y_i|}
    """
    cdef inline DTYPE_t dist(self, DTYPE_t* x1, DTYPE_t* x2, ITYPE_t size):
        cdef DTYPE_t denom, d = 0
        for j in range(size):
            denom = abs(x1[j]) + abs(x2[j])
            if denom > 0:
                d += abs(x1[j] - x2[j]) / denom
        return d


#------------------------------------------------------------
# Bray-Curtis Distance
#  D(x, y) = sum[abs(x_i - y_i)] / sum[abs(x_i) + abs(y_i)]
cdef class BrayCurtisDistance(DistanceMetric):
    """Bray-Curtis Distance

    Bray-Curtis distance is meant for discrete-valued vectors, though it is
    a valid metric for real-valued vectors.

    .. math::
       D(x, y) = \frac{\sum_i |x_i - y_i|}{\sum_i(|x_i| + |y_i|)}
    """
    cdef inline DTYPE_t dist(self, DTYPE_t* x1, DTYPE_t* x2, ITYPE_t size):
        cdef DTYPE_t num = 0, denom = 0
        for j in range(size):
            num += abs(x1[j] - x2[j])
            denom += abs(x1[j]) + abs(x2[j])
        if denom > 0:
            return num / denom
        else:
            return 0.0


#------------------------------------------------------------
# Jaccard Distance (boolean)
#  D(x, y) = N_unequal(x, y) / N_nonzero(x, y)
cdef class JaccardDistance(DistanceMetric):
    """Jaccard Distance

    Jaccard Distance is a dissimilarity measure for boolean-valued
    vectors. All nonzero entries will be treated as True, zero entries will
    be treated as False.

    .. math::
       D(x, y) = \frac{N_{TF} + N_{FT}}{N_{TT} + N_{TF} + N_{FT}}
    """
    cdef inline DTYPE_t dist(self, DTYPE_t* x1, DTYPE_t* x2, ITYPE_t size):
        cdef int tf1, tf2, n_eq = 0, nnz = 0
        for j in range(size):
            tf1 = x1[j] != 0
            tf2 = x2[j] != 0
            nnz += (tf1 or tf2)
            n_eq += (tf1 and tf2)
        return (nnz - n_eq) * 1.0 / nnz


#------------------------------------------------------------
# Matching Distance (boolean)
#  D(x, y) = n_neq / n
cdef class MatchingDistance(DistanceMetric):
    """Matching Distance

    Matching Distance is a dissimilarity measure for boolean-valued
    vectors. All nonzero entries will be treated as True, zero entries will
    be treated as False.

    .. math::
       D(x, y) = \frac{N_{TF} + N_{FT}}{N}
    """
    cdef inline DTYPE_t dist(self, DTYPE_t* x1, DTYPE_t* x2, ITYPE_t size):
        cdef int tf1, tf2, n_neq = 0
        for j in range(size):
            tf1 = x1[j] != 0
            tf2 = x2[j] != 0
            n_neq += (tf1 != tf2)
        return n_neq * 1. / size


#------------------------------------------------------------
# Dice Distance (boolean)
#  D(x, y) = n_neq / (2 * ntt + n_neq)
cdef class DiceDistance(DistanceMetric):
    """Dice Distance

    Dice Distance is a dissimilarity measure for boolean-valued
    vectors. All nonzero entries will be treated as True, zero entries will
    be treated as False.

    .. math::
       D(x, y) = \frac{N_{TF} + N_{FT}}{2 * N_{TT} + N_{TF} + N_{FT}}
    """
    cdef inline DTYPE_t dist(self, DTYPE_t* x1, DTYPE_t* x2, ITYPE_t size):
        cdef int tf1, tf2, n_neq = 0, ntt = 0
        for j in range(size):
            tf1 = x1[j] != 0
            tf2 = x2[j] != 0
            ntt += (tf1 and tf2)
            n_neq += (tf1 != tf2)
        return n_neq / (2.0 * ntt + n_neq)


#------------------------------------------------------------
# Kulsinski Distance (boolean)
#  D(x, y) = (ntf + nft - ntt + n) / (n_neq + n)
cdef class KulsinskiDistance(DistanceMetric):
    """Kulsinski Distance

    Kulsinski Distance is a dissimilarity measure for boolean-valued
    vectors. All nonzero entries will be treated as True, zero entries will
    be treated as False.

    .. math::
       D(x, y) = 1 - \frac{N_{TT}}{N + N_{TF} + N_{FT}}
    """
    cdef inline DTYPE_t dist(self, DTYPE_t* x1, DTYPE_t* x2, ITYPE_t size):
        cdef int tf1, tf2, ntt = 0, n_neq = 0
        for j in range(size):
            tf1 = x1[j] != 0
            tf2 = x2[j] != 0
            n_neq += (tf1 != tf2)
            ntt += (tf1 and tf2)
        return (n_neq - ntt + size) * 1.0 / (n_neq + size)


#------------------------------------------------------------
# Roger-Stanimoto Distance (boolean)
#  D(x, y) = 2 * n_neq / (n + n_neq)
cdef class RogerStanimotoDistance(DistanceMetric):
    """Roger-Stanimoto Distance

    Roger-Stanimoto Distance is a dissimilarity measure for boolean-valued
    vectors. All nonzero entries will be treated as True, zero entries will
    be treated as False.

    .. math::
       D(x, y) = \frac{2 (N_{TF} + N_{FT})}{N + N_{TF} + N_{FT}}
    """
    cdef inline DTYPE_t dist(self, DTYPE_t* x1, DTYPE_t* x2, ITYPE_t size):
        cdef int tf1, tf2, n_neq = 0
        for j in range(size):
            tf1 = x1[j] != 0
            tf2 = x2[j] != 0
            n_neq += (tf1 != tf2)
        return (2.0 * n_neq) / (size + n_neq)


#------------------------------------------------------------
# Russell-Rao Distance (boolean)
#  D(x, y) = (n - ntt) / n
cdef class RussellRaoDistance(DistanceMetric):
    """Russell-Rao Distance

    Russell-Rao Distance is a dissimilarity measure for boolean-valued
    vectors. All nonzero entries will be treated as True, zero entries will
    be treated as False.

    .. math::
       D(x, y) = \frac{N - N_{TT}}{N}
    """
    cdef inline DTYPE_t dist(self, DTYPE_t* x1, DTYPE_t* x2, ITYPE_t size):
        cdef int tf1, tf2, ntt = 0
        for j in range(size):
            tf1 = x1[j] != 0
            tf2 = x2[j] != 0
            ntt += (tf1 and tf2)
        return (size - ntt) * 1. / size


#------------------------------------------------------------
# Sokal-Michener Distance (boolean)
#  D(x, y) = 2 * n_neq / (n + n_neq)
cdef class SokalMichenerDistance(DistanceMetric):
    """Sokal-Michener Distance

    Sokal-Michener Distance is a dissimilarity measure for boolean-valued
    vectors. All nonzero entries will be treated as True, zero entries will
    be treated as False.

    .. math::
       D(x, y) = \frac{2 (N_{TF} + N_{FT})}{N + N_{TF} + N_{FT}}
    """
    cdef inline DTYPE_t dist(self, DTYPE_t* x1, DTYPE_t* x2, ITYPE_t size):
        cdef int tf1, tf2, n_neq = 0
        for j in range(size):
            tf1 = x1[j] != 0
            tf2 = x2[j] != 0
            n_neq += (tf1 != tf2)
        return (2.0 * n_neq) / (size + n_neq)


#------------------------------------------------------------
# Sokal-Sneath Distance (boolean)
#  D(x, y) = n_neq / (0.5 * n_tt + n_neq)
cdef class SokalSneathDistance(DistanceMetric):
    """Sokal-Sneath Distance

    Sokal-Sneath Distance is a dissimilarity measure for boolean-valued
    vectors. All nonzero entries will be treated as True, zero entries will
    be treated as False.

    .. math::
       D(x, y) = \frac{N_{TF} + N_{FT}}{N_{TT} / 2 + N_{TF} + N_{FT}}
    """
    cdef inline DTYPE_t dist(self, DTYPE_t* x1, DTYPE_t* x2, ITYPE_t size):
        cdef int tf1, tf2, ntt = 0, n_neq = 0
        for j in range(size):
            tf1 = x1[j] != 0
            tf2 = x2[j] != 0
            n_neq += (tf1 != tf2)
            ntt += (tf1 and tf2)
        return n_neq / (0.5 * ntt + n_neq)


#------------------------------------------------------------
# Yule Distance (boolean)
#  D(x, y) = 2 * ntf * nft / (ntt * nff + ntf * nft)
# [This is not a true metric, so we will leave it out.]
#
#cdef class YuleDistance(DistanceMetric):
#    cdef inline DTYPE_t dist(self, DTYPE_t* x1, DTYPE_t* x2, ITYPE_t size):
#        cdef int tf1, tf2, ntf = 0, nft = 0, ntt = 0, nff = 0
#        for j in range(size):
#            tf1 = x1[j] != 0
#            tf2 = x2[j] != 0
#            ntt += tf1 and tf2
#            ntf += tf1 and (tf2 == 0)
#            nft += (tf1 == 0) and tf2
#        nff = size - ntt - ntf - nft
#        return (2.0 * ntf * nft) / (ntt * nff + ntf * nft)


#------------------------------------------------------------
# Cosine Distance
#  D(x, y) = dot(x, y) / (|x| * |y|)
# [This is not a true metric, so we will leave it out.]
#
#cdef class CosineDistance(DistanceMetric):
#    cdef inline DTYPE_t dist(self, DTYPE_t* x1, DTYPE_t* x2, ITYPE_t size):
#        cdef DTYPE_t d = 0, norm1 = 0, norm2 = 0
#        for j in range(size):
#            d += x1[j] * x2[j]
#            norm1 += x1[j] * x1[j]
#            norm2 += x2[j] * x2[j]
#        return 1.0 - d / sqrt(norm1 * norm2)


#------------------------------------------------------------
# Correlation Distance
#  D(x, y) = dot((x - mx), (y - my)) / (|x - mx| * |y - my|)
# [This is not a true metric, so we will leave it out.]
#
#cdef class CorrelationDistance(DistanceMetric):
#    cdef inline DTYPE_t dist(self, DTYPE_t* x1, DTYPE_t* x2, ITYPE_t size):
#        cdef DTYPE_t mu1 = 0, mu2 = 0, x1nrm = 0, x2nrm = 0, x1Tx2 = 0
#        cdef DTYPE_t tmp1, tmp2
#
#        for i in range(size):
#            mu1 += x1[i]
#            mu2 += x2[i]
#        mu1 /= size
#        mu2 /= size
#
#        for i in range(size):
#            tmp1 = x1[i] - mu1
#            tmp2 = x2[i] - mu2
#            x1nrm += tmp1 * tmp1
#            x2nrm += tmp2 * tmp2
#            x1Tx2 += tmp1 * tmp2
#
#        return (1. - x1Tx2) / sqrt(x1nrm * x2nrm)


######################################################################
# Functions for benchmarking and testing
cdef DTYPE_t[:, ::1] euclidean_cdist(DTYPE_t[:, ::1] X, DTYPE_t[:, ::1] Y):
    if X.shape[1] != Y.shape[1]:
        raise ValueError('X and Y must have the same second dimension')

    cdef DTYPE_t[:, ::1] D = np.zeros((X.shape[0], Y.shape[0]), dtype=DTYPE)

    for i1 in range(X.shape[0]):
        for i2 in range(Y.shape[0]):
            D[i1, i2] = euclidean_dist(&X[i1, 0], &Y[i2, 0], X.shape[1])
    return D

def euclidean_pairwise_inline(DTYPE_t[:, ::1] X, DTYPE_t[:, ::1] Y):
    D = euclidean_cdist(X, Y)
    return np.asarray(D)


def euclidean_pairwise_class(DTYPE_t[:, ::1] X, DTYPE_t[:, ::1] Y):
    cdef EuclideanDistance eucl_dist = EuclideanDistance()

    assert X.shape[1] == Y.shape[1]
    cdef DTYPE_t[:, ::1] D = np.zeros((X.shape[0], Y.shape[0]), dtype=DTYPE)

    for i1 in range(X.shape[0]):
        for i2 in range(Y.shape[0]):
            D[i1, i2] = eucl_dist.dist(&X[i1, 0], &Y[i2, 0], X.shape[1])
    return np.asarray(D)


def euclidean_pairwise_polymorphic(DTYPE_t[:, ::1] X, DTYPE_t[:, ::1] Y):
    cdef DistanceMetric eucl_dist = EuclideanDistance()

    assert X.shape[1] == Y.shape[1]
    cdef DTYPE_t[:, ::1] D = np.zeros((X.shape[0], Y.shape[0]), dtype=DTYPE)

    for i1 in range(X.shape[0]):
        for i2 in range(Y.shape[0]):
            D[i1, i2] = eucl_dist.dist(&X[i1, 0], &Y[i2, 0], X.shape[1])
    return np.asarray(D)

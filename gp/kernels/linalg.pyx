#cython: boundscheck=False

from __future__ import division

import numpy as np
cimport numpy as np

cdef log = np.log
cdef exp = np.exp
cdef sqrt = np.sqrt

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef DTYPE_t EPS = np.finfo(DTYPE).eps
cdef DTYPE_t MIN = log(np.exp2(DTYPE(np.finfo(DTYPE).minexp + 4)))


def safe_mul(DTYPE_t x, DTYPE_t y):
    cdef DTYPE_t v, out
    if x == 0 or y == 0:
        out = 0
    else:
        v = log(x) + log(y)
        if v < MIN:
            out = 0
        else:
            out = exp(v)
    return out


def cholesky(np.ndarray[DTYPE_t, ndim=2] A, np.ndarray[DTYPE_t, ndim=2] L):
    """Cholesky decomposition of a symmetric positive-definite matrix,
    implemented for the numpy.float64 dtype.

    A = L * L.T
    Only L (the lower part) is returned.

    Based on `mpmath` implementation of the cholesky decomposition.

    """
    cdef int n, m, n2, m2, i, j
    cdef DTYPE_t v

    n = A.shape[0]
    m = A.shape[1]
    if n != m:
        raise ValueError("need n-by-n matrix")

    n2 = L.shape[0]
    m2 = L.shape[1]
    if n != n2 or m != m2:
        raise ValueError("shape mismatch for L")

    L.fill(0)

    for j in xrange(n):
        v = 0
        for k in xrange(j):
            v += safe_mul(L[j, k], L[j, k])

        s = A[j, j] - v
        if s < EPS:
            raise ValueError("matrix is not positive definite")

        L[j, j] = sqrt(s)
        for i in xrange(j+1, n):
            v = 0
            for k in xrange(j):
                v += safe_mul(L[i, k], L[j, k])

            L[i, j] = (A[i, j] - v) / L[j, j]


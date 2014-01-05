# cython: boundscheck=False
# cython: wraparound=False

from __future__ import division

import numpy as np
cimport numpy as np

from libc.math cimport log, exp, sqrt, M_PI

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef DTYPE_t SQRT_2_DIV_PI = sqrt(2.0 / M_PI)
cdef DTYPE_t MIN = log(np.exp2(DTYPE(np.finfo(DTYPE).minexp + 4)))


def K(np.ndarray[DTYPE_t, mode='c', ndim=2] out, np.ndarray[DTYPE_t, mode='c', ndim=1] x1, np.ndarray[DTYPE_t, mode='c', ndim=1] x2, DTYPE_t h, DTYPE_t w):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef DTYPE_t h2, w2, c1, c2, e

    h2 = h ** 2
    w2 = w ** 2

    c1 = -0.5/w2
    c2 = 0.5*SQRT_2_DIV_PI*h2/w

    for i in xrange(x1_s):
        for j in xrange(x2_s):
            e = c1*(x1[i] - x2[j])**2
            if e < MIN:
                out[i, j] = 0
            else:
                out[i, j] = c2*exp(e)


def jacobian(np.ndarray[DTYPE_t, mode='c', ndim=3] out, np.ndarray[DTYPE_t, mode='c', ndim=1] x1, np.ndarray[DTYPE_t, mode='c', ndim=1] x2, DTYPE_t h, DTYPE_t w):
    dK_dh(out[0], x1, x2, h, w)
    dK_dw(out[1], x1, x2, h, w)


def hessian(np.ndarray[DTYPE_t, mode='c', ndim=4] out, np.ndarray[DTYPE_t, mode='c', ndim=1] x1, np.ndarray[DTYPE_t, mode='c', ndim=1] x2, DTYPE_t h, DTYPE_t w):
    d2K_dhdh(out[0, 0], x1, x2, h, w)
    d2K_dhdw(out[0, 1], x1, x2, h, w)
    d2K_dwdh(out[1, 0], x1, x2, h, w)
    d2K_dwdw(out[1, 1], x1, x2, h, w)


def dK_dh(np.ndarray[DTYPE_t, mode='c', ndim=2] out, np.ndarray[DTYPE_t, mode='c', ndim=1] x1, np.ndarray[DTYPE_t, mode='c', ndim=1] x2, DTYPE_t h, DTYPE_t w):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef DTYPE_t h2, w2, c1, c2, e

    h2 = h ** 2
    w2 = w ** 2

    c1 = -0.5/w2
    c2 = SQRT_2_DIV_PI*h/w

    for i in xrange(x1_s):
        for j in xrange(x2_s):
            e = c1*(x1[i] - x2[j])**2
            if e < MIN:
                out[i, j] = 0
            else:
                out[i, j] = c2*exp(e)


def dK_dw(np.ndarray[DTYPE_t, mode='c', ndim=2] out, np.ndarray[DTYPE_t, mode='c', ndim=1] x1, np.ndarray[DTYPE_t, mode='c', ndim=1] x2, DTYPE_t h, DTYPE_t w):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef DTYPE_t h2, w2, c1, c2, c3, d2, e

    h2 = h ** 2
    w2 = w ** 2

    c1 = -0.5/w2
    c2 = 0.5*SQRT_2_DIV_PI*h2/w2
    c3 = 0.5*SQRT_2_DIV_PI*h2/w**4

    for i in xrange(x1_s):
        for j in xrange(x2_s):
            d2 = (x1[i] - x2[j])**2
            e = c1*d2
            if e < MIN:
                out[i, j] = 0
            else:
                out[i, j] = exp(e)*(c3*d2 - c2)


def d2K_dhdh(np.ndarray[DTYPE_t, mode='c', ndim=2] out, np.ndarray[DTYPE_t, mode='c', ndim=1] x1, np.ndarray[DTYPE_t, mode='c', ndim=1] x2, DTYPE_t h, DTYPE_t w):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef DTYPE_t h2, w2, c1, c2, e

    h2 = h ** 2
    w2 = w ** 2

    c1 = -0.5/w2
    c2 = SQRT_2_DIV_PI/w

    for i in xrange(x1_s):
        for j in xrange(x2_s):
            e = c1*(x1[i] - x2[j])**2
            if e < MIN:
                out[i, j] = 0
            else:
                out[i, j] = c2*exp(e)


def d2K_dhdw(np.ndarray[DTYPE_t, mode='c', ndim=2] out, np.ndarray[DTYPE_t, mode='c', ndim=1] x1, np.ndarray[DTYPE_t, mode='c', ndim=1] x2, DTYPE_t h, DTYPE_t w):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef DTYPE_t h2, w2, c1, c2, c3, d2, e

    h2 = h ** 2
    w2 = w ** 2

    c1 = -0.5/w2
    c2 = SQRT_2_DIV_PI*h/w2
    c3 = SQRT_2_DIV_PI*h/w**4

    for i in xrange(x1_s):
        for j in xrange(x2_s):
            d2 = (x1[i] - x2[j])**2
            e = c1*d2
            if e < MIN:
                out[i, j] = 0
            else:
                out[i, j] = exp(e)*(c3*d2 - c2)


def d2K_dwdh(np.ndarray[DTYPE_t, mode='c', ndim=2] out, np.ndarray[DTYPE_t, mode='c', ndim=1] x1, np.ndarray[DTYPE_t, mode='c', ndim=1] x2, DTYPE_t h, DTYPE_t w):
    d2K_dhdw(out, x1, x2, h, w)


def d2K_dwdw(np.ndarray[DTYPE_t, mode='c', ndim=2] out, np.ndarray[DTYPE_t, mode='c', ndim=1] x1, np.ndarray[DTYPE_t, mode='c', ndim=1] x2, DTYPE_t h, DTYPE_t w):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef DTYPE_t h2, w2, c1, c2, c3, c4, d2, e

    h2 = h ** 2
    w2 = w ** 2

    c1 = -0.5/w2
    c2 = SQRT_2_DIV_PI*h2/w**3
    c3 = 2.5*SQRT_2_DIV_PI*h2/w**5
    c4 = 0.5*SQRT_2_DIV_PI*h2/w**7

    for i in xrange(x1_s):
        for j in xrange(x2_s):
            d2 = (x1[i] - x2[j])**2
            e = c1*d2
            if e < MIN:
                out[i, j] = 0
            else:
                out[i, j] = exp(e)*(c4*d2**2 - c3*d2 + c2)

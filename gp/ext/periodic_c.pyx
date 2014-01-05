# cython: boundscheck=False
# cython: wraparound=False

from __future__ import division

import numpy as np
cimport numpy as np

from libc.math cimport log, exp, sqrt, cos, sin, M_PI

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef DTYPE_t SQRT_2_DIV_PI = sqrt(2.0 / M_PI)
cdef DTYPE_t MIN = log(np.exp2(DTYPE(np.finfo(DTYPE).minexp + 4)))


def K(np.ndarray[DTYPE_t, mode='c', ndim=2] out, np.ndarray[DTYPE_t, mode='c', ndim=1] x1, np.ndarray[DTYPE_t, mode='c', ndim=1] x2, DTYPE_t h, DTYPE_t w, DTYPE_t p):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef DTYPE_t hw, w2, d

    h2 = h ** 2
    w2 = w ** 2

    for i in xrange(x1_s):
        for j in xrange(x2_s):
            d = x1[i] - x2[j]
            out[i, j] = h2*exp(-2.0*sin(0.5*d/p)**2/w2)


def jacobian(np.ndarray[DTYPE_t, mode='c', ndim=3] out, np.ndarray[DTYPE_t, mode='c', ndim=1] x1, np.ndarray[DTYPE_t, mode='c', ndim=1] x2, DTYPE_t h, DTYPE_t w, DTYPE_t p):
    dK_dh(out[0], x1, x2, h, w, p)
    dK_dw(out[1], x1, x2, h, w, p)
    dK_dp(out[2], x1, x2, h, w, p)


def hessian(np.ndarray[DTYPE_t, mode='c', ndim=4] out, np.ndarray[DTYPE_t, mode='c', ndim=1] x1, np.ndarray[DTYPE_t, mode='c', ndim=1] x2, DTYPE_t h, DTYPE_t w, DTYPE_t p):
    d2K_dhdh(out[0, 0], x1, x2, h, w, p)
    d2K_dhdw(out[0, 1], x1, x2, h, w, p)
    d2K_dhdp(out[0, 2], x1, x2, h, w, p)

    d2K_dwdh(out[1, 0], x1, x2, h, w, p)
    d2K_dwdw(out[1, 1], x1, x2, h, w, p)
    d2K_dwdp(out[1, 2], x1, x2, h, w, p)

    d2K_dpdh(out[2, 0], x1, x2, h, w, p)
    d2K_dpdw(out[2, 1], x1, x2, h, w, p)
    d2K_dpdp(out[2, 2], x1, x2, h, w, p)


def dK_dh(np.ndarray[DTYPE_t, mode='c', ndim=2] out, np.ndarray[DTYPE_t, mode='c', ndim=1] x1, np.ndarray[DTYPE_t, mode='c', ndim=1] x2, DTYPE_t h, DTYPE_t w, DTYPE_t p):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef DTYPE_t hw, w2, d

    h2 = h ** 2
    w2 = w ** 2

    for i in xrange(x1_s):
        for j in xrange(x2_s):
            d = x1[i] - x2[j]
            out[i, j] = 2.0*h*exp(-2.0*sin(0.5*d/p)**2/w2)


def dK_dw(np.ndarray[DTYPE_t, mode='c', ndim=2] out, np.ndarray[DTYPE_t, mode='c', ndim=1] x1, np.ndarray[DTYPE_t, mode='c', ndim=1] x2, DTYPE_t h, DTYPE_t w, DTYPE_t p):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef DTYPE_t hw, w2, d

    h2 = h ** 2
    w2 = w ** 2

    for i in xrange(x1_s):
        for j in xrange(x2_s):
            d = x1[i] - x2[j]
            out[i, j] = 4.0*h2*exp(-2.0*sin(0.5*d/p)**2/w2)*sin(0.5*d/p)**2/w**3


def dK_dp(np.ndarray[DTYPE_t, mode='c', ndim=2] out, np.ndarray[DTYPE_t, mode='c', ndim=1] x1, np.ndarray[DTYPE_t, mode='c', ndim=1] x2, DTYPE_t h, DTYPE_t w, DTYPE_t p):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef DTYPE_t hw, w2, p2, d

    h2 = h ** 2
    w2 = w ** 2
    p2 = p ** 2

    for i in xrange(x1_s):
        for j in xrange(x2_s):
            d = x1[i] - x2[j]
            out[i, j] = 2.0*d*h2*exp(-2.0*sin(0.5*d/p)**2/w2)*sin(0.5*d/p)*cos(0.5*d/p)/(p2*w2)


def d2K_dhdh(np.ndarray[DTYPE_t, mode='c', ndim=2] out, np.ndarray[DTYPE_t, mode='c', ndim=1] x1, np.ndarray[DTYPE_t, mode='c', ndim=1] x2, DTYPE_t h, DTYPE_t w, DTYPE_t p):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef DTYPE_t hw, w2, d

    h2 = h ** 2
    w2 = w ** 2

    for i in xrange(x1_s):
        for j in xrange(x2_s):
            d = x1[i] - x2[j]
            out[i, j] = 2.0*exp(-2.0*sin(0.5*d/p)**2/w2)


def d2K_dhdw(np.ndarray[DTYPE_t, mode='c', ndim=2] out, np.ndarray[DTYPE_t, mode='c', ndim=1] x1, np.ndarray[DTYPE_t, mode='c', ndim=1] x2, DTYPE_t h, DTYPE_t w, DTYPE_t p):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef DTYPE_t hw, w2, d

    h2 = h ** 2
    w2 = w ** 2

    for i in xrange(x1_s):
        for j in xrange(x2_s):
            d = x1[i] - x2[j]
            out[i, j] = 8.0*h*exp(-2.0*sin(0.5*d/p)**2/w2)*sin(0.5*d/p)**2/w**3


def d2K_dhdp(np.ndarray[DTYPE_t, mode='c', ndim=2] out, np.ndarray[DTYPE_t, mode='c', ndim=1] x1, np.ndarray[DTYPE_t, mode='c', ndim=1] x2, DTYPE_t h, DTYPE_t w, DTYPE_t p):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef DTYPE_t hw, w2, p2, d

    h2 = h ** 2
    w2 = w ** 2
    p2 = p ** 2

    for i in xrange(x1_s):
        for j in xrange(x2_s):
            d = x1[i] - x2[j]
            out[i, j] = 4.0*d*h*exp(-2.0*sin(0.5*d/p)**2/w2)*sin(0.5*d/p)*cos(0.5*d/p)/(p2*w2)


def d2K_dwdh(np.ndarray[DTYPE_t, mode='c', ndim=2] out, np.ndarray[DTYPE_t, mode='c', ndim=1] x1, np.ndarray[DTYPE_t, mode='c', ndim=1] x2, DTYPE_t h, DTYPE_t w, DTYPE_t p):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef DTYPE_t hw, w2, d

    h2 = h ** 2
    w2 = w ** 2

    for i in xrange(x1_s):
        for j in xrange(x2_s):
            d = x1[i] - x2[j]
            out[i, j] = 8.0*h*exp(-2.0*sin(0.5*d/p)**2/w2)*sin(0.5*d/p)**2/w**3


def d2K_dwdw(np.ndarray[DTYPE_t, mode='c', ndim=2] out, np.ndarray[DTYPE_t, mode='c', ndim=1] x1, np.ndarray[DTYPE_t, mode='c', ndim=1] x2, DTYPE_t h, DTYPE_t w, DTYPE_t p):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef DTYPE_t hw, w2, d

    h2 = h ** 2
    w2 = w ** 2

    for i in xrange(x1_s):
        for j in xrange(x2_s):
            d = x1[i] - x2[j]
            out[i, j] = -12.0*h2*exp(-2.0*sin(0.5*d/p)**2/w2)*sin(0.5*d/p)**2/w**4 + 16.0*h2*exp(-2.0*sin(0.5*d/p)**2/w2)*sin(0.5*d/p)**4/w**6


def d2K_dwdp(np.ndarray[DTYPE_t, mode='c', ndim=2] out, np.ndarray[DTYPE_t, mode='c', ndim=1] x1, np.ndarray[DTYPE_t, mode='c', ndim=1] x2, DTYPE_t h, DTYPE_t w, DTYPE_t p):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef DTYPE_t hw, w2, p2, d

    h2 = h ** 2
    w2 = w ** 2
    p2 = p ** 2

    for i in xrange(x1_s):
        for j in xrange(x2_s):
            d = x1[i] - x2[j]
            out[i, j] = -4.0*d*h2*exp(-2.0*sin(0.5*d/p)**2/w2)*sin(0.5*d/p)*cos(0.5*d/p)/(p2*w**3) + 8.0*d*h2*exp(-2.0*sin(0.5*d/p)**2/w2)*sin(0.5*d/p)**3*cos(0.5*d/p)/(p2*w**5)


def d2K_dpdh(np.ndarray[DTYPE_t, mode='c', ndim=2] out, np.ndarray[DTYPE_t, mode='c', ndim=1] x1, np.ndarray[DTYPE_t, mode='c', ndim=1] x2, DTYPE_t h, DTYPE_t w, DTYPE_t p):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef DTYPE_t hw, w2, p2, d

    h2 = h ** 2
    w2 = w ** 2
    p2 = p ** 2

    for i in xrange(x1_s):
        for j in xrange(x2_s):
            d = x1[i] - x2[j]
            out[i, j] = 4.0*d*h*exp(-2.0*sin(0.5*d/p)**2/w2)*sin(0.5*d/p)*cos(0.5*d/p)/(p2*w2)


def d2K_dpdw(np.ndarray[DTYPE_t, mode='c', ndim=2] out, np.ndarray[DTYPE_t, mode='c', ndim=1] x1, np.ndarray[DTYPE_t, mode='c', ndim=1] x2, DTYPE_t h, DTYPE_t w, DTYPE_t p):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef DTYPE_t hw, w2, p2, d

    h2 = h ** 2
    w2 = w ** 2
    p2 = p ** 2

    for i in xrange(x1_s):
        for j in xrange(x2_s):
            d = x1[i] - x2[j]
            out[i, j] = -4.0*d*h2*exp(-2.0*sin(0.5*d/p)**2/w2)*sin(0.5*d/p)*cos(0.5*d/p)/(p2*w**3) + 8.0*d*h2*exp(-2.0*sin(0.5*d/p)**2/w2)*sin(0.5*d/p)**3*cos(0.5*d/p)/(p2*w**5)


def d2K_dpdp(np.ndarray[DTYPE_t, mode='c', ndim=2] out, np.ndarray[DTYPE_t, mode='c', ndim=1] x1, np.ndarray[DTYPE_t, mode='c', ndim=1] x2, DTYPE_t h, DTYPE_t w, DTYPE_t p):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef DTYPE_t hw, w2, d

    h2 = h ** 2
    w2 = w ** 2

    for i in xrange(x1_s):
        for j in xrange(x2_s):
            d = x1[i] - x2[j]
            out[i, j] = d**2*h2*exp(-2.0*sin(0.5*d/p)**2/w2)*sin(0.5*d/p)**2/(p**4*w2) - 1.0*d**2*h2*exp(-2.0*sin(0.5*d/p)**2/w2)*cos(0.5*d/p)**2/(p**4*w2) + 4.0*d**2*h2*exp(-2.0*sin(0.5*d/p)**2/w2)*sin(0.5*d/p)**2*cos(0.5*d/p)**2/(p**4*w**4) - 4.0*d*h2*exp(-2.0*sin(0.5*d/p)**2/w2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**3*w2)



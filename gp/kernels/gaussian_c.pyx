from __future__ import division

import numpy as np
cimport numpy as np

from math import sqrt, exp, pi

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

def K(np.ndarray[DTYPE_t, ndim=2] out, np.ndarray[DTYPE_t, ndim=1] x1, np.ndarray[DTYPE_t, ndim=1] x2, float h, float w):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef float d
    for i in xrange(x1_s):
        for j in xrange(x2_s):
            d = x1[i] - x2[j]
            out[i, j] = 0.5*sqrt(2)*h**2*exp(-0.5*d**2/w**2)/(sqrt(pi)*sqrt(w**2))


def jacobian(np.ndarray[DTYPE_t, ndim=3] out, np.ndarray[DTYPE_t, ndim=1] x1, np.ndarray[DTYPE_t, ndim=1] x2, float h, float w):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef float d
    for i in xrange(x1_s):
        for j in xrange(x2_s):
            d = x1[i] - x2[j]
            out[0, i, j] = sqrt(2)*h*exp(-0.5*d**2/w**2)/(sqrt(pi)*sqrt(w**2))
            out[1, i, j] = 0.5*sqrt(2)*d**2*h**2*exp(-0.5*d**2/w**2)/(sqrt(pi)*w**3*sqrt(w**2)) - 0.5*sqrt(2)*h**2*exp(-0.5*d**2/w**2)/(sqrt(pi)*w*sqrt(w**2))


def hessian(np.ndarray[DTYPE_t, ndim=4] out, np.ndarray[DTYPE_t, ndim=1] x1, np.ndarray[DTYPE_t, ndim=1] x2, float h, float w):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef float d
    for i in xrange(x1_s):
        for j in xrange(x2_s):
            d = x1[i] - x2[j]
            # h
            out[0, 0, i, j] = sqrt(2)*exp(-0.5*d**2/w**2)/(sqrt(pi)*sqrt(w**2))
            out[0, 1, i, j] = sqrt(2)*d**2*h*exp(-0.5*d**2/w**2)/(sqrt(pi)*w**3*sqrt(w**2)) - sqrt(2)*h*exp(-0.5*d**2/w**2)/(sqrt(pi)*w*sqrt(w**2))
            # w
            out[1, 0, i, j] = sqrt(2)*d**2*h*exp(-0.5*d**2/w**2)/(sqrt(pi)*w**3*sqrt(w**2)) - sqrt(2)*h*exp(-0.5*d**2/w**2)/(sqrt(pi)*w*sqrt(w**2))
            out[1, 1, i, j] = 0.5*sqrt(2)*d**4*h**2*exp(-0.5*d**2/w**2)/(sqrt(pi)*w**6*sqrt(w**2)) - 2.5*sqrt(2)*d**2*h**2*exp(-0.5*d**2/w**2)/(sqrt(pi)*w**4*sqrt(w**2)) + sqrt(2)*h**2*exp(-0.5*d**2/w**2)/(sqrt(pi)*w**2*sqrt(w**2))


def dK_dh(np.ndarray[DTYPE_t, ndim=2] out, np.ndarray[DTYPE_t, ndim=1] x1, np.ndarray[DTYPE_t, ndim=1] x2, float h, float w):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef float d
    for i in xrange(x1_s):
        for j in xrange(x2_s):
            d = x1[i] - x2[j]
            out[i, j] = sqrt(2)*h*exp(-0.5*d**2/w**2)/(sqrt(pi)*sqrt(w**2))


def dK_dw(np.ndarray[DTYPE_t, ndim=2] out, np.ndarray[DTYPE_t, ndim=1] x1, np.ndarray[DTYPE_t, ndim=1] x2, float h, float w):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef float d
    for i in xrange(x1_s):
        for j in xrange(x2_s):
            d = x1[i] - x2[j]
            out[i, j] = 0.5*sqrt(2)*d**2*h**2*exp(-0.5*d**2/w**2)/(sqrt(pi)*w**3*sqrt(w**2)) - 0.5*sqrt(2)*h**2*exp(-0.5*d**2/w**2)/(sqrt(pi)*w*sqrt(w**2))


def d2K_dhdh(np.ndarray[DTYPE_t, ndim=2] out, np.ndarray[DTYPE_t, ndim=1] x1, np.ndarray[DTYPE_t, ndim=1] x2, float h, float w):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef float d
    for i in xrange(x1_s):
        for j in xrange(x2_s):
            d = x1[i] - x2[j]
            out[i, j] = sqrt(2)*exp(-0.5*d**2/w**2)/(sqrt(pi)*sqrt(w**2))


def d2K_dhdw(np.ndarray[DTYPE_t, ndim=2] out, np.ndarray[DTYPE_t, ndim=1] x1, np.ndarray[DTYPE_t, ndim=1] x2, float h, float w):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef float d
    for i in xrange(x1_s):
        for j in xrange(x2_s):
            d = x1[i] - x2[j]
            out[i, j] = sqrt(2)*d**2*h*exp(-0.5*d**2/w**2)/(sqrt(pi)*w**3*sqrt(w**2)) - sqrt(2)*h*exp(-0.5*d**2/w**2)/(sqrt(pi)*w*sqrt(w**2))


def d2K_dwdh(np.ndarray[DTYPE_t, ndim=2] out, np.ndarray[DTYPE_t, ndim=1] x1, np.ndarray[DTYPE_t, ndim=1] x2, float h, float w):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef float d
    for i in xrange(x1_s):
        for j in xrange(x2_s):
            d = x1[i] - x2[j]
            out[i, j] = sqrt(2)*d**2*h*exp(-0.5*d**2/w**2)/(sqrt(pi)*w**3*sqrt(w**2)) - sqrt(2)*h*exp(-0.5*d**2/w**2)/(sqrt(pi)*w*sqrt(w**2))


def d2K_dwdw(np.ndarray[DTYPE_t, ndim=2] out, np.ndarray[DTYPE_t, ndim=1] x1, np.ndarray[DTYPE_t, ndim=1] x2, float h, float w):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef float d
    for i in xrange(x1_s):
        for j in xrange(x2_s):
            d = x1[i] - x2[j]
            out[i, j] = 0.5*sqrt(2)*d**4*h**2*exp(-0.5*d**2/w**2)/(sqrt(pi)*w**6*sqrt(w**2)) - 2.5*sqrt(2)*d**2*h**2*exp(-0.5*d**2/w**2)/(sqrt(pi)*w**4*sqrt(w**2)) + sqrt(2)*h**2*exp(-0.5*d**2/w**2)/(sqrt(pi)*w**2*sqrt(w**2))



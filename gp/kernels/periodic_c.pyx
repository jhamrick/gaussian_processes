from __future__ import division

import numpy as np
cimport numpy as np

from math import sqrt, exp, pi, sin, cos

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

def K(np.ndarray[DTYPE_t, ndim=2] out, np.ndarray[DTYPE_t, ndim=1] x1, np.ndarray[DTYPE_t, ndim=1] x2, double h, double w, double p):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef double d
    for i in xrange(x1_s):
        for j in xrange(x2_s):
            d = x1[i] - x2[j]
            out[i, j] = h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)


def jacobian(np.ndarray[DTYPE_t, ndim=3] out, np.ndarray[DTYPE_t, ndim=1] x1, np.ndarray[DTYPE_t, ndim=1] x2, double h, double w, double p):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef double d
    for i in xrange(x1_s):
        for j in xrange(x2_s):
            d = x1[i] - x2[j]
            out[0, i, j] = 2.0*h*exp(-2.0*sin(0.5*d/p)**2/w**2)
            out[1, i, j] = 4.0*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2/w**3
            out[2, i, j] = 2.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**2*w**2)


def hessian(np.ndarray[DTYPE_t, ndim=4] out, np.ndarray[DTYPE_t, ndim=1] x1, np.ndarray[DTYPE_t, ndim=1] x2, double h, double w, double p):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef double d
    for i in xrange(x1_s):
        for j in xrange(x2_s):
            d = x1[i] - x2[j]
            # h
            out[0, 0, i, j] = 2.0*exp(-2.0*sin(0.5*d/p)**2/w**2)
            out[0, 1, i, j] = 8.0*h*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2/w**3
            out[0, 2, i, j] = 4.0*d*h*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**2*w**2)
            # w
            out[1, 0, i, j] = 8.0*h*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2/w**3
            out[1, 1, i, j] = -12.0*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2/w**4 + 16.0*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**4/w**6
            out[1, 2, i, j] = -4.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**2*w**3) + 8.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**3*cos(0.5*d/p)/(p**2*w**5)
            # p
            out[2, 0, i, j] = 4.0*d*h*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**2*w**2)
            out[2, 1, i, j] = -4.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**2*w**3) + 8.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**3*cos(0.5*d/p)/(p**2*w**5)
            out[2, 2, i, j]= d**2*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2/(p**4*w**2) - 1.0*d**2*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*cos(0.5*d/p)**2/(p**4*w**2) + 4.0*d**2*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2*cos(0.5*d/p)**2/(p**4*w**4) - 4.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**3*w**2)


def dK_dh(np.ndarray[DTYPE_t, ndim=2] out, np.ndarray[DTYPE_t, ndim=1] x1, np.ndarray[DTYPE_t, ndim=1] x2, double h, double w, double p):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef double d
    for i in xrange(x1_s):
        for j in xrange(x2_s):
            d = x1[i] - x2[j]
            out[i, j] = 2.0*h*exp(-2.0*sin(0.5*d/p)**2/w**2)


def dK_dw(np.ndarray[DTYPE_t, ndim=2] out, np.ndarray[DTYPE_t, ndim=1] x1, np.ndarray[DTYPE_t, ndim=1] x2, double h, double w, double p):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef double d
    for i in xrange(x1_s):
        for j in xrange(x2_s):
            d = x1[i] - x2[j]
            out[i, j] = 4.0*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2/w**3


def dK_dp(np.ndarray[DTYPE_t, ndim=2] out, np.ndarray[DTYPE_t, ndim=1] x1, np.ndarray[DTYPE_t, ndim=1] x2, double h, double w, double p):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef double d
    for i in xrange(x1_s):
        for j in xrange(x2_s):
            d = x1[i] - x2[j]
            out[i, j] = 2.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**2*w**2)


def d2K_dhdh(np.ndarray[DTYPE_t, ndim=2] out, np.ndarray[DTYPE_t, ndim=1] x1, np.ndarray[DTYPE_t, ndim=1] x2, double h, double w, double p):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef double d
    for i in xrange(x1_s):
        for j in xrange(x2_s):
            d = x1[i] - x2[j]
            out[i, j] = 2.0*exp(-2.0*sin(0.5*d/p)**2/w**2)


def d2K_dhdw(np.ndarray[DTYPE_t, ndim=2] out, np.ndarray[DTYPE_t, ndim=1] x1, np.ndarray[DTYPE_t, ndim=1] x2, double h, double w, double p):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef double d
    for i in xrange(x1_s):
        for j in xrange(x2_s):
            d = x1[i] - x2[j]
            out[i, j] = 8.0*h*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2/w**3


def d2K_dhdp(np.ndarray[DTYPE_t, ndim=2] out, np.ndarray[DTYPE_t, ndim=1] x1, np.ndarray[DTYPE_t, ndim=1] x2, double h, double w, double p):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef double d
    for i in xrange(x1_s):
        for j in xrange(x2_s):
            d = x1[i] - x2[j]
            out[i, j] = 4.0*d*h*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**2*w**2)


def d2K_dwdh(np.ndarray[DTYPE_t, ndim=2] out, np.ndarray[DTYPE_t, ndim=1] x1, np.ndarray[DTYPE_t, ndim=1] x2, double h, double w, double p):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef double d
    for i in xrange(x1_s):
        for j in xrange(x2_s):
            d = x1[i] - x2[j]
            out[i, j] = 8.0*h*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2/w**3


def d2K_dwdw(np.ndarray[DTYPE_t, ndim=2] out, np.ndarray[DTYPE_t, ndim=1] x1, np.ndarray[DTYPE_t, ndim=1] x2, double h, double w, double p):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef double d
    for i in xrange(x1_s):
        for j in xrange(x2_s):
            d = x1[i] - x2[j]
            out[i, j] = -12.0*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2/w**4 + 16.0*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**4/w**6


def d2K_dwdp(np.ndarray[DTYPE_t, ndim=2] out, np.ndarray[DTYPE_t, ndim=1] x1, np.ndarray[DTYPE_t, ndim=1] x2, double h, double w, double p):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef double d
    for i in xrange(x1_s):
        for j in xrange(x2_s):
            d = x1[i] - x2[j]
            out[i, j] = -4.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**2*w**3) + 8.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**3*cos(0.5*d/p)/(p**2*w**5)


def d2K_dpdh(np.ndarray[DTYPE_t, ndim=2] out, np.ndarray[DTYPE_t, ndim=1] x1, np.ndarray[DTYPE_t, ndim=1] x2, double h, double w, double p):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef double d
    for i in xrange(x1_s):
        for j in xrange(x2_s):
            d = x1[i] - x2[j]
            out[i, j] = 4.0*d*h*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**2*w**2)


def d2K_dpdw(np.ndarray[DTYPE_t, ndim=2] out, np.ndarray[DTYPE_t, ndim=1] x1, np.ndarray[DTYPE_t, ndim=1] x2, double h, double w, double p):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef double d
    for i in xrange(x1_s):
        for j in xrange(x2_s):
            d = x1[i] - x2[j]
            out[i, j] = -4.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**2*w**3) + 8.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**3*cos(0.5*d/p)/(p**2*w**5)


def d2K_dpdp(np.ndarray[DTYPE_t, ndim=2] out, np.ndarray[DTYPE_t, ndim=1] x1, np.ndarray[DTYPE_t, ndim=1] x2, double h, double w, double p):
    cdef int x1_s = x1.size
    cdef int x2_s = x2.size
    cdef int i, j
    cdef double d
    for i in xrange(x1_s):
        for j in xrange(x2_s):
            d = x1[i] - x2[j]
            out[i, j] = d**2*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2/(p**4*w**2) - 1.0*d**2*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*cos(0.5*d/p)**2/(p**4*w**2) + 4.0*d**2*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2*cos(0.5*d/p)**2/(p**4*w**4) - 4.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**3*w**2)



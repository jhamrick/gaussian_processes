# cython: boundscheck=False
# cython: wraparound=False

from __future__ import division

import numpy as np
cimport numpy as np

from libc.math cimport log, exp, sqrt, M_PI, INFINITY

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef DTYPE_t MIN = log(np.exp2(DTYPE(np.finfo(DTYPE).minexp + 4)))


def log_lh(np.ndarray[DTYPE_t, mode='c', ndim=1] y, np.ndarray[DTYPE_t, mode='c', ndim=2] K, np.ndarray[DTYPE_t, mode='c', ndim=1] Kiy):
    cdef int sign
    cdef DTYPE_t logdet, data_fit, complexity_penalty, constant, llh

    sign, logdet = np.linalg.slogdet(K)
    if sign != 1:
        llh = -INFINITY

    else:
        data_fit = -0.5 * DTYPE(np.dot(y, Kiy))
        complexity_penalty = -0.5 * logdet
        constant = -0.5 * y.size * log(2 * M_PI)
        llh = data_fit + complexity_penalty + constant

    return llh


def dloglh_dtheta(np.ndarray[DTYPE_t, mode='c', ndim=1] y, np.ndarray[DTYPE_t, mode='c', ndim=2] Ki, np.ndarray[DTYPE_t, mode='c', ndim=3] Kj, np.ndarray[DTYPE_t, mode='c', ndim=1] Kiy, DTYPE_t s, np.ndarray[DTYPE_t, mode='c', ndim=1] dloglh):
    cdef int n = Kj.shape[0]
    cdef int m = Kj.shape[1]
    cdef int i
    cdef np.ndarray[DTYPE_t, mode='c', ndim=2] k = np.empty((m, m), dtype=DTYPE)
    cdef DTYPE_t t0, t1

    for i in xrange(n+1):
        if i < n:
            k[:] = np.dot(Ki, Kj[i])
        else:
            k[:] = np.dot(Ki, np.eye(m) * 2 * s)

        t0 = 0.5 * np.dot(y, np.dot(k, Kiy))
        t1 = -0.5 * np.trace(k)
        dloglh[i] = t0 + t1


def dlh_dtheta(np.ndarray[DTYPE_t, mode='c', ndim=1] y, np.ndarray[DTYPE_t, mode='c', ndim=2] Ki, np.ndarray[DTYPE_t, mode='c', ndim=3] Kj, np.ndarray[DTYPE_t, mode='c', ndim=1] Kiy, DTYPE_t s, DTYPE_t lh, np.ndarray[DTYPE_t, mode='c', ndim=1] dlh):
    cdef int n = Kj.shape[0]
    cdef int m = Kj.shape[1]
    cdef int i
    cdef np.ndarray[DTYPE_t, mode='c', ndim=2] k = np.empty((m, m), dtype=DTYPE)
    cdef DTYPE_t t0, t1

    for i in xrange(n+1):
        if i < n:
            k[:] = np.dot(Ki, Kj[i])
        else:
            k[:] = np.dot(Ki, np.eye(m) * 2 * s)
        
        t0 = np.dot(y, np.dot(k, Kiy))
        t1 = np.trace(k)
        dlh[i] = 0.5 * lh * (t0 - t1)


def d2lh_dtheta2(np.ndarray[DTYPE_t, mode='c', ndim=1] y, np.ndarray[DTYPE_t, mode='c', ndim=2] Ki, np.ndarray[DTYPE_t, mode='c', ndim=3] Kj, np.ndarray[DTYPE_t, mode='c', ndim=4] Kh, np.ndarray[DTYPE_t, mode='c', ndim=1] Kiy, DTYPE_t s, DTYPE_t lh, np.ndarray[DTYPE_t, mode='c', ndim=1] dlh, np.ndarray[DTYPE_t, mode='c', ndim=2] d2lh):
    cdef int n = Kj.shape[0]
    cdef int m = Kj.shape[1]
    cdef int i, j
    cdef np.ndarray[DTYPE_t, mode='c', ndim=3] dK = np.empty((n+1, m, m), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, mode='c', ndim=3] dKi = np.empty((n+1, m, m), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, mode='c', ndim=2] d2k = np.empty((m, m), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, mode='c', ndim=2] KidK_i = np.empty((m, m), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, mode='c', ndim=2] dKi_jdK_i = np.empty((m, m), dtype=DTYPE)
    cdef DTYPE_t ydKi_iy, ydKi_iy_tr, t0, t1, t1a, t1b, t1c

    for i in xrange(n+1):
        # first kernel derivatives
        if i < n:
            dK[i] = Kj[i]
        else:
            dK[i] = np.eye(m) * 2 * s
        
        dKi[i] = np.dot(-Ki, np.dot(dK[i], Ki))

    for i in xrange(n+1):
        KidK_i[:] = np.dot(Ki, dK[i])
        ydKi_iy = np.dot(y, np.dot(KidK_i, Kiy))
        ydKi_iy_tr = ydKi_iy - np.trace(KidK_i)

        for j in xrange(n+1):
            # second kernel derivatives
            if j < n and i < n:
                d2k[:] = Kh[i, j]
            elif j == n and i == n:
                d2k[:] = np.eye(m) * 2
            else:
                d2k.fill(0)

            dKi_jdK_i[:] = np.dot(dKi[j], dK[i])
            t0 = dlh[j] * ydKi_iy_tr

            t1a = np.dot(y, np.dot(dKi_jdK_i, Kiy))
            t1b = np.dot(Kiy, np.dot(d2k, Kiy))
            t1c = np.dot(Kiy, np.dot(dK[i], np.dot(dKi[j], y)))
            t1 = lh * (t1a + t1b + t1c - np.trace(dKi_jdK_i + np.dot(Ki, d2k)))
            d2lh[i, j] = 0.5 * (t0 + t1)


def dm_dtheta(np.ndarray[DTYPE_t, mode='c', ndim=1] y, np.ndarray[DTYPE_t, mode='c', ndim=2] Ki, np.ndarray[DTYPE_t, mode='c', ndim=3] Kj, np.ndarray[DTYPE_t, mode='c', ndim=3] Kjxo, np.ndarray[DTYPE_t, mode='c', ndim=2] Kxox, DTYPE_t s, np.ndarray[DTYPE_t, mode='c', ndim=2] dm):
    cdef int n = Kj.shape[0]
    cdef int m = Kj.shape[1]
    cdef int m2 = Kjxo.shape[1]
    cdef int i
    cdef np.ndarray[DTYPE_t, mode='c', ndim=2] dKxox_dtheta = np.empty((m2, m), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, mode='c', ndim=2] dKxx_dtheta = np.empty((m, m), dtype=DTYPE)

    for i in xrange(n+1):
        if i < n:
            dKxox_dtheta[:] = Kjxo[i]
            dKxx_dtheta[:] = Kj[i]
        else:
            dKxox_dtheta.fill(0)
            dKxx_dtheta[:] = np.eye(m) * 2 * s

        dm[i] = np.dot(dKxox_dtheta, np.dot(Ki, y))
        dm[i] -= np.dot(Kxox, np.dot(np.dot(Ki, np.dot(dKxx_dtheta, Ki)), y))

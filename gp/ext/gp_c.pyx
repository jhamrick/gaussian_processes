#cython: boundscheck=False

from __future__ import division

import numpy as np
cimport numpy as np

cdef log = np.log
cdef exp = np.exp
cdef sqrt = np.sqrt
cdef dot = np.dot
cdef slogdet = np.linalg.slogdet
cdef trace = np.trace
cdef eye = np.eye

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef DTYPE_t MIN = log(np.exp2(DTYPE(np.finfo(DTYPE).minexp + 4)))


def log_lh(np.ndarray[DTYPE_t, ndim=1] y, np.ndarray[DTYPE_t, ndim=2] K, np.ndarray[DTYPE_t, ndim=1] Kiy):
    cdef int sign
    cdef DTYPE_t logdet, data_fit, complexity_penalty, constant, llh

    sign, logdet = slogdet(K)
    if sign != 1:
        llh = -np.inf

    else:
        data_fit = -0.5 * DTYPE(dot(y, Kiy))
        complexity_penalty = -0.5 * logdet
        constant = -0.5 * y.size * log(2 * np.pi)
        llh = data_fit + complexity_penalty + constant

    return llh


def dloglh_dtheta(np.ndarray[DTYPE_t, ndim=1] y, np.ndarray[DTYPE_t, ndim=2] Ki, np.ndarray[DTYPE_t, ndim=3] Kj, np.ndarray[DTYPE_t, ndim=1] Kiy, DTYPE_t s, np.ndarray[DTYPE_t, ndim=1] dloglh):
    cdef int n = Kj.shape[0]
    cdef int m = Kj.shape[1]
    cdef int i
    cdef np.ndarray[DTYPE_t, ndim=2] k = np.empty((m, m), dtype=DTYPE)
    cdef DTYPE_t t0, t1

    for i in xrange(n+1):
        if i < n:
            k[:] = dot(Ki, Kj[i])
        else:
            k[:] = dot(Ki, eye(m) * 2 * s)

        t0 = 0.5 * dot(y, dot(k, Kiy))
        t1 = -0.5 * trace(k)
        dloglh[i] = t0 + t1


def dlh_dtheta(np.ndarray[DTYPE_t, ndim=1] y, np.ndarray[DTYPE_t, ndim=2] Ki, np.ndarray[DTYPE_t, ndim=3] Kj, np.ndarray[DTYPE_t, ndim=1] Kiy, DTYPE_t s, DTYPE_t lh, np.ndarray[DTYPE_t, ndim=1] dlh):
    cdef int n = Kj.shape[0]
    cdef int m = Kj.shape[1]
    cdef int i
    cdef np.ndarray[DTYPE_t, ndim=2] k = np.empty((m, m), dtype=DTYPE)
    cdef DTYPE_t t0, t1

    for i in xrange(n+1):
        if i < n:
            k[:] = dot(Ki, Kj[i])
        else:
            k[:] = dot(Ki, eye(m) * 2 * s)
        
        t0 = dot(y, dot(k, Kiy))
        t1 = trace(k)
        dlh[i] = 0.5 * lh * (t0 - t1)


def d2lh_dtheta2(np.ndarray[DTYPE_t, ndim=1] y, np.ndarray[DTYPE_t, ndim=2] Ki, np.ndarray[DTYPE_t, ndim=3] Kj, np.ndarray[DTYPE_t, ndim=4] Kh, np.ndarray[DTYPE_t, ndim=1] Kiy, DTYPE_t s, DTYPE_t lh, np.ndarray[DTYPE_t, ndim=1] dlh, np.ndarray[DTYPE_t, ndim=2] d2lh):
    cdef int n = Kj.shape[0]
    cdef int m = Kj.shape[1]
    cdef int i, j
    cdef np.ndarray[DTYPE_t, ndim=3] dK = np.empty((n+1, m, m), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] dKi = np.empty((n+1, m, m), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] d2k = np.empty((m, m), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] KidK_i = np.empty((m, m), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] dKi_jdK_i = np.empty((m, m), dtype=DTYPE)
    cdef DTYPE_t ydKi_iy, ydKi_iy_tr, t0, t1, t1a, t1b, t1c

    for i in xrange(n+1):
        # first kernel derivatives
        if i < n:
            dK[i] = dot(Ki, Kj[i])
        else:
            dK[i] = dot(Ki, eye(m) * 2 * s)
        
        dKi[i] = dot(-Ki, dot(dK[i], Ki))

    for i in xrange(n+1):
        KidK_i[:] = dot(Ki, dK[i])
        ydKi_iy = dot(y, dot(KidK_i, Kiy))
        ydKi_iy_tr = ydKi_iy - trace(KidK_i)

        for j in xrange(n+1):
            # second kernel derivatives
            if j < n and i < n:
                d2k[:] = Kh[i, j]
            elif j == n and i == n:
                d2k[:] = eye(m) * 2
            else:
                d2k.fill(0)

            dKi_jdK_i[:] = dot(dKi[j], dK[i])
            t0 = dlh[j] * ydKi_iy_tr

            t1a = dot(y, dot(dKi_jdK_i, Kiy))
            t1b = dot(Kiy, dot(d2k, Kiy))
            t1c = dot(Kiy, dot(dK[i], dot(dKi[j], y)))
            t1 = lh * (t1a + t1b + t1c - trace(dKi_jdK_i + dot(Ki, d2k)))
            d2lh[i, j] = 0.5 * (t0 + t1)

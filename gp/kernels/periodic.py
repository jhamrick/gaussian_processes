__all__ = ['PeriodicKernel']

import numpy as np
import sympy as sym

from functools import wraps
from gp.ext import periodic_c
from . import Kernel

DTYPE = np.float64
EPS = np.finfo(DTYPE).eps


class PeriodicKernel(Kernel):
    r"""
    Periodic kernel function.

    Parameters
    ----------
    h : float
        Output scale kernel parameter
    w : float
        Input scale kernel parameter
    p : float
        Period kernel parameter

    Notes
    -----
    The periodic kernel is defined by Equation 4.31 of [RW06]_:

    .. math:: K(x_1, x_2) = h^2\exp\left(\frac{-2\sin^2\left(\frac{x_1-x_2}{2p}\right)}{w^2}\right)

    where :math:`w` is the input scale parameter (equivalent to the
    standard deviation of the Gaussian), :math:`h` is the output
    scale parameter, and :math:`p` is the period kernel parameter.

    """

    def __init__(self, h, w, p):
        if h < EPS:
            raise ValueError("invalid value for h: %s" % h)
        if w < EPS:
            raise ValueError("invalid value for w: %s" % w)
        if p < EPS:
            raise ValueError("invalid value for p: %s" % w)

        self.h = DTYPE(h) #: Output scale kernel parameter
        self.w = DTYPE(w) #: Input scale kernel parameter
        self.p = DTYPE(p) #: Period kernel parameter

    @property
    def params(self):
        r"""
        Kernel parameters.

        Returns
        -------
        params : numpy.ndarray ``(h, w, p)``

        """
        return np.array([self.h, self.w, self.p], dtype=DTYPE)

    @params.setter
    def params(self, val):
        h, w, p = val

        if h < EPS:
            raise ValueError("invalid value for h: %s" % h)
        if w < EPS:
            raise ValueError("invalid value for w: %s" % w)
        if p < EPS:
            raise ValueError("invalid value for p: %s" % w)

        self.h = DTYPE(h)
        self.w = DTYPE(w)
        self.p = DTYPE(p)

    @property
    @wraps(Kernel.sym_K)
    def sym_K(self):
        h = sym.Symbol('h')
        w = sym.Symbol('w')
        p = sym.Symbol('p')
        d = sym.Symbol('d')

        h2 = h ** 2
        w2 = w ** 2

        f = h2 * sym.exp(-2. * (sym.sin(d / (2. * p)) ** 2) / w2)
        return f

    @wraps(Kernel.K)
    def K(self, x1, x2, out=None):
        if out is None:
            out = np.empty((x1.size, x2.size), dtype=DTYPE)
        periodic_c.K(out, x1, x2, self.h, self.w, self.p)
        return out

    @wraps(Kernel.jacobian)
    def jacobian(self, x1, x2, out=None):
        if out is None:
            out = np.empty((3, x1.size, x2.size), dtype=DTYPE)
        periodic_c.jacobian(out, x1, x2, self.h, self.w, self.p)
        return out

    @wraps(Kernel.hessian)
    def hessian(self, x1, x2, out=None):
        if out is None:
            out = np.empty((3, 3, x1.size, x2.size), dtype=DTYPE)
        periodic_c.hessian(out, x1, x2, self.h, self.w, self.p)
        return out

    def dK_dh(self, x1, x2, out=None):
        if out is None:
            out = np.empty((x1.size, x2.size), dtype=DTYPE)
        periodic_c.dK_dh(out, x1, x2, self.h, self.w, self.p)
        return out

    def dK_dw(self, x1, x2, out=None):
        if out is None:
            out = np.empty((x1.size, x2.size), dtype=DTYPE)
        periodic_c.dK_dw(out, x1, x2, self.h, self.w, self.p)
        return out

    def dK_dp(self, x1, x2, out=None):
        if out is None:
            out = np.empty((x1.size, x2.size), dtype=DTYPE)
        periodic_c.dK_dp(out, x1, x2, self.h, self.w, self.p)
        return out

    def d2K_dhdh(self, x1, x2, out=None):
        if out is None:
            out = np.empty((x1.size, x2.size), dtype=DTYPE)
        periodic_c.d2K_dhdh(out, x1, x2, self.h, self.w, self.p)
        return out

    def d2K_dhdw(self, x1, x2, out=None):
        if out is None:
            out = np.empty((x1.size, x2.size), dtype=DTYPE)
        periodic_c.d2K_dhdw(out, x1, x2, self.h, self.w, self.p)
        return out

    def d2K_dhdp(self, x1, x2, out=None):
        if out is None:
            out = np.empty((x1.size, x2.size), dtype=DTYPE)
        periodic_c.d2K_dhdp(out, x1, x2, self.h, self.w, self.p)
        return out

    def d2K_dwdh(self, x1, x2, out=None):
        if out is None:
            out = np.empty((x1.size, x2.size), dtype=DTYPE)
        periodic_c.d2K_dwdh(out, x1, x2, self.h, self.w, self.p)
        return out

    def d2K_dwdw(self, x1, x2, out=None):
        if out is None:
            out = np.empty((x1.size, x2.size), dtype=DTYPE)
        periodic_c.d2K_dwdw(out, x1, x2, self.h, self.w, self.p)
        return out

    def d2K_dwdp(self, x1, x2, out=None):
        if out is None:
            out = np.empty((x1.size, x2.size), dtype=DTYPE)
        periodic_c.d2K_dwdp(out, x1, x2, self.h, self.w, self.p)
        return out

    def d2K_dpdh(self, x1, x2, out=None):
        if out is None:
            out = np.empty((x1.size, x2.size), dtype=DTYPE)
        periodic_c.d2K_dpdh(out, x1, x2, self.h, self.w, self.p)
        return out

    def d2K_dpdw(self, x1, x2, out=None):
        if out is None:
            out = np.empty((x1.size, x2.size), dtype=DTYPE)
        periodic_c.d2K_dpdw(out, x1, x2, self.h, self.w, self.p)
        return out

    def d2K_dpdp(self, x1, x2, out=None):
        if out is None:
            out = np.empty((x1.size, x2.size), dtype=DTYPE)
        periodic_c.d2K_dpdp(out, x1, x2, self.h, self.w, self.p)
        return out

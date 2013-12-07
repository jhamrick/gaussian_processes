__all__ = ['GaussianKernel']

import numpy as np
import sympy as sym

from functools import wraps
from gp.ext import gaussian_c
from . import Kernel

DTYPE = np.float64
EPS = np.finfo(DTYPE).eps


class GaussianKernel(Kernel):
    r"""
    Gaussian kernel function.

    Parameters
    ----------
    h : float
        Output scale kernel parameter
    w : float
        Input scale kernel parameter

    Notes
    -----
    The Gaussian kernel is defined as:

    .. math:: K(x_1, x_2) = \frac{h^2}{\sqrt{2\pi w^2}}\exp\left(-\frac{(x_1-x_2)^2}{2w^2}\right),

    where :math:`w` is the input scale parameter (equivalent to the
    standard deviation of the Gaussian) and :math:`h` is the output
    scale parameter.

    """

    def __init__(self, h, w):
        if h < EPS:
            raise ValueError("invalid value for h: %s" % h)
        if w < EPS:
            raise ValueError("invalid value for w: %s" % w)

        self.h = DTYPE(h) #: Output scale kernel parameter
        self.w = DTYPE(w) #: Input scale kernel parameter

    @property
    def params(self):
        r"""
        Kernel parameters.

        Returns
        -------
        params : numpy.ndarray ``(h, w)``

        """
        return np.array([self.h, self.w], dtype=DTYPE)

    @params.setter
    def params(self, val):
        h, w = val

        if h < EPS:
            raise ValueError("invalid value for h: %s" % h)
        if w < EPS:
            raise ValueError("invalid value for w: %s" % w)

        self.h = DTYPE(h)
        self.w = DTYPE(w)

    @property
    @wraps(Kernel.sym_K)
    def sym_K(self):
        h = sym.Symbol('h')
        w = sym.Symbol('w')
        d = sym.Symbol('d')

        h2 = h ** 2
        w2 = w ** 2
        d2 = d ** 2

        f = h2 * (1. / sym.sqrt(2 * sym.pi * w2)) * sym.exp(-d2 / (2.0 * w2))
        return f

    @wraps(Kernel.K)
    def K(self, x1, x2, out=None):
        if out is None:
            out = np.empty((x1.size, x2.size), dtype=DTYPE)
        gaussian_c.K(out, x1, x2, self.h, self.w)
        return out

    @wraps(Kernel.jacobian)
    def jacobian(self, x1, x2, out=None):
        if out is None:
            out = np.empty((2, x1.size, x2.size), dtype=DTYPE)
        gaussian_c.jacobian(out, x1, x2, self.h, self.w)
        return out

    @wraps(Kernel.hessian)
    def hessian(self, x1, x2, out=None):
        if out is None:
            out = np.empty((2, 2, x1.size, x2.size), dtype=DTYPE)
        gaussian_c.hessian(out, x1, x2, self.h, self.w)
        return out

    def dK_dh(self, x1, x2, out=None):
        if out is None:
            out = np.empty((x1.size, x2.size), dtype=DTYPE)
        gaussian_c.dK_dh(out, x1, x2, self.h, self.w)
        return out

    def dK_dw(self, x1, x2, out=None):
        if out is None:
            out = np.empty((x1.size, x2.size), dtype=DTYPE)
        gaussian_c.dK_dw(out, x1, x2, self.h, self.w)
        return out

    def d2K_dhdh(self, x1, x2, out=None):
        if out is None:
            out = np.empty((x1.size, x2.size), dtype=DTYPE)
        gaussian_c.d2K_dhdh(out, x1, x2, self.h, self.w)
        return out

    def d2K_dhdw(self, x1, x2, out=None):
        if out is None:
            out = np.empty((x1.size, x2.size), dtype=DTYPE)
        gaussian_c.d2K_dhdw(out, x1, x2, self.h, self.w)
        return out

    def d2K_dwdh(self, x1, x2, out=None):
        if out is None:
            out = np.empty((x1.size, x2.size), dtype=DTYPE)
        gaussian_c.d2K_dwdh(out, x1, x2, self.h, self.w)
        return out

    def d2K_dwdw(self, x1, x2, out=None):
        if out is None:
            out = np.empty((x1.size, x2.size), dtype=DTYPE)
        gaussian_c.d2K_dwdw(out, x1, x2, self.h, self.w)
        return out

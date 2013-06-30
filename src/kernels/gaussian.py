__all__ = ['GaussianKernel']

import numpy as np
import sympy as sym

from numpy import exp, sqrt, pi
from functools import wraps
from . import Kernel

from util import staticlazyjit
staticlazyjit_mat = staticlazyjit(
    'f8[:,:](f8[:], f8[:], f8, f8)', warn=False)


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
        self.h = h #: Output scale kernel parameter
        self.w = w #: Input scale kernel parameter

    @property
    def params(self):
        r"""
        Kernel parameters.

        Returns
        -------
        params : tuple ``(h, w)``

        """
        return (self.h, self.w)

    @params.setter
    def params(self, val):
        self.h, self.w = val

    @property
    @wraps(Kernel.sym_K)
    def sym_K(self):
        h = sym.Symbol('h')
        w = sym.Symbol('w')
        d = sym.Symbol('d')

        h2 = h ** 2
        w2 = w ** 2
        d2 = d ** 2

        f = h2 * (1. / sym.sqrt(2*sym.pi*w2)) * sym.exp(-d2 / (2.0 * w2))
        return f

    @staticlazyjit_mat
    @wraps(Kernel._K)
    def _K(x1, x2, h, w):
        Kxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                Kxx[i, j] = 0.5*sqrt(2)*h**2*exp(-0.5*d**2/w**2)/(sqrt(pi)*sqrt(w**2))
        return Kxx

    @staticlazyjit('f8[:,:,:](f8[:], f8[:], f8, f8)', warn=False)
    @wraps(Kernel._jacobian)
    def _jacobian(x1, x2, h, w):
        dKxx = np.empty((2, x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[0, i, j] = sqrt(2)*h*exp(-0.5*d**2/w**2)/(sqrt(pi)*sqrt(w**2))
                dKxx[1, i, j] = 0.5*sqrt(2)*d**2*h**2*exp(-0.5*d**2/w**2)/(sqrt(pi)*w**3*sqrt(w**2)) - 0.5*sqrt(2)*h**2*exp(-0.5*d**2/w**2)/(sqrt(pi)*w*sqrt(w**2))
        return dKxx

    @staticlazyjit('f8[:,:,:](f8[:], f8[:], f8, f8)', warn=False)
    @wraps(Kernel._hessian)
    def _hessian(x1, x2, h, w):
        dKxx = np.empty((2, 2, x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                # h
                dKxx[0, 0, i, j] = sqrt(2)*exp(-0.5*d**2/w**2)/(sqrt(pi)*sqrt(w**2))
                dKxx[0, 1, i, j] = sqrt(2)*d**2*h*exp(-0.5*d**2/w**2)/(sqrt(pi)*w**3*sqrt(w**2)) - sqrt(2)*h*exp(-0.5*d**2/w**2)/(sqrt(pi)*w*sqrt(w**2))
                # w
                dKxx[1, 0, i, j] = sqrt(2)*d**2*h*exp(-0.5*d**2/w**2)/(sqrt(pi)*w**3*sqrt(w**2)) - sqrt(2)*h*exp(-0.5*d**2/w**2)/(sqrt(pi)*w*sqrt(w**2))
                dKxx[1, 1, i, j] = 0.5*sqrt(2)*d**4*h**2*exp(-0.5*d**2/w**2)/(sqrt(pi)*w**6*sqrt(w**2)) - 2.5*sqrt(2)*d**2*h**2*exp(-0.5*d**2/w**2)/(sqrt(pi)*w**4*sqrt(w**2)) + sqrt(2)*h**2*exp(-0.5*d**2/w**2)/(sqrt(pi)*w**2*sqrt(w**2))

        return dKxx

    @staticlazyjit_mat
    def _dK_dh(x1, x2, h, w):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = sqrt(2)*h*exp(-0.5*d**2/w**2)/(sqrt(pi)*sqrt(w**2))
        return dKxx

    @staticlazyjit_mat
    def _dK_dw(x1, x2, h, w):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = 0.5*sqrt(2)*d**2*h**2*exp(-0.5*d**2/w**2)/(sqrt(pi)*w**3*sqrt(w**2)) - 0.5*sqrt(2)*h**2*exp(-0.5*d**2/w**2)/(sqrt(pi)*w*sqrt(w**2))
        return dKxx

    @staticlazyjit_mat
    def _d2K_dhdh(x1, x2, h, w):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = sqrt(2)*exp(-0.5*d**2/w**2)/(sqrt(pi)*sqrt(w**2))
        return dKxx

    @staticlazyjit_mat
    def _d2K_dhdw(x1, x2, h, w):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = sqrt(2)*d**2*h*exp(-0.5*d**2/w**2)/(sqrt(pi)*w**3*sqrt(w**2)) - sqrt(2)*h*exp(-0.5*d**2/w**2)/(sqrt(pi)*w*sqrt(w**2))
        return dKxx

    @staticlazyjit_mat
    def _d2K_dwdh(x1, x2, h, w):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = sqrt(2)*d**2*h*exp(-0.5*d**2/w**2)/(sqrt(pi)*w**3*sqrt(w**2)) - sqrt(2)*h*exp(-0.5*d**2/w**2)/(sqrt(pi)*w*sqrt(w**2))
        return dKxx

    @staticlazyjit_mat
    def _d2K_dwdw(x1, x2, h, w):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = 0.5*sqrt(2)*d**4*h**2*exp(-0.5*d**2/w**2)/(sqrt(pi)*w**6*sqrt(w**2)) - 2.5*sqrt(2)*d**2*h**2*exp(-0.5*d**2/w**2)/(sqrt(pi)*w**4*sqrt(w**2)) + sqrt(2)*h**2*exp(-0.5*d**2/w**2)/(sqrt(pi)*w**2*sqrt(w**2))
        return dKxx

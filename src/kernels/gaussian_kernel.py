import numpy as np
from numpy import exp, sqrt, pi

import sympy as sym

from base_kernel import BaseKernel, lazyjit


class GaussianKernel(object):
    """Represents a gaussian kernel function, of the form:

    $$k(x_1, x_2) = h^2\frac{1}{\sqrt{2\pi w^2}}\exp(-\frac{(x_1-x_2)^2}{2w^2})$$

    References
    ----------
    Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian processes
        for machine learning. MIT Press.

    """

    __metaclass__ = BaseKernel
    __slots__ = ['h', 'w']

    _sym_h = sym.Symbol('h')
    _sym_w = sym.Symbol('w')
    _sym_d = sym.Symbol('d')

    def __init__(self, h, w):
        """Create a GaussianKernel object at specific parameter values.

        Parameters
        ----------
        h : number
            Output scale kernel parameter
        w : number
            Input scale (Gaussian standard deviation) kernel parameter

        """

        self.h = h
        self.w = w

    @property
    def sym_K(self):
        h = self._sym_h
        w = self._sym_w
        d = self._sym_d

        h2 = h ** 2
        w2 = w ** 2
        d2 = d ** 2

        f = h2 * (1. / sym.sqrt(2*sym.pi*w2)) * sym.exp(-d2 / (2.0 * w2))
        return f

    def copy(self):
        return GaussianKernel(*self.params)

    @property
    def params(self):
        return (self.h, self.w)

    @params.setter
    def params(self, val):
        self.h, self.w = val

    @staticmethod
    @lazyjit('f8[:,:](f8[:], f8[:], f8, f8)', warn=False)
    def _K(x1, x2, h, w):
        Kxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                Kxx[i, j] = 0.5*sqrt(2)*h**2*exp(-0.5*d**2/w**2)/(sqrt(pi)*sqrt(w**2))
        return Kxx

    @staticmethod
    @lazyjit('f8[:,:](f8[:], f8[:], f8, f8)', warn=False)
    def _dK_dh(x1, x2, h, w):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = sqrt(2)*h*exp(-0.5*d**2/w**2)/(sqrt(pi)*sqrt(w**2))
        return dKxx

    @staticmethod
    @lazyjit('f8[:,:](f8[:], f8[:], f8, f8)', warn=False)
    def _dK_dw(x1, x2, h, w):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = 0.5*sqrt(2)*d**2*h**2*exp(-0.5*d**2/w**2)/(sqrt(pi)*w**3*sqrt(w**2)) - 0.5*sqrt(2)*h**2*exp(-0.5*d**2/w**2)/(sqrt(pi)*w*sqrt(w**2))
        return dKxx

    @staticmethod
    @lazyjit('f8[:,:,:](f8[:], f8[:], f8, f8)', warn=False)
    def _jacobian(x1, x2, h, w):
        dKxx = np.empty((2, x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[0, i, j] = sqrt(2)*h*exp(-0.5*d**2/w**2)/(sqrt(pi)*sqrt(w**2))
                dKxx[1, i, j] = 0.5*sqrt(2)*d**2*h**2*exp(-0.5*d**2/w**2)/(sqrt(pi)*w**3*sqrt(w**2)) - 0.5*sqrt(2)*h**2*exp(-0.5*d**2/w**2)/(sqrt(pi)*w*sqrt(w**2))
        return dKxx

    @staticmethod
    @lazyjit('f8[:,:](f8[:], f8[:], f8, f8)', warn=False)
    def _d2K_dhdh(x1, x2, h, w):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = sqrt(2)*exp(-0.5*d**2/w**2)/(sqrt(pi)*sqrt(w**2))
        return dKxx

    @staticmethod
    @lazyjit('f8[:,:](f8[:], f8[:], f8, f8)', warn=False)
    def _d2K_dhdw(x1, x2, h, w):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = sqrt(2)*d**2*h*exp(-0.5*d**2/w**2)/(sqrt(pi)*w**3*sqrt(w**2)) - sqrt(2)*h*exp(-0.5*d**2/w**2)/(sqrt(pi)*w*sqrt(w**2))
        return dKxx

    @staticmethod
    @lazyjit('f8[:,:](f8[:], f8[:], f8, f8)', warn=False)
    def _d2K_dwdh(x1, x2, h, w):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = sqrt(2)*d**2*h*exp(-0.5*d**2/w**2)/(sqrt(pi)*w**3*sqrt(w**2)) - sqrt(2)*h*exp(-0.5*d**2/w**2)/(sqrt(pi)*w*sqrt(w**2))
        return dKxx

    @staticmethod
    @lazyjit('f8[:,:](f8[:], f8[:], f8, f8)', warn=False)
    def _d2K_dwdw(x1, x2, h, w):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = 0.5*sqrt(2)*d**4*h**2*exp(-0.5*d**2/w**2)/(sqrt(pi)*w**6*sqrt(w**2)) - 2.5*sqrt(2)*d**2*h**2*exp(-0.5*d**2/w**2)/(sqrt(pi)*w**4*sqrt(w**2)) + sqrt(2)*h**2*exp(-0.5*d**2/w**2)/(sqrt(pi)*w**2*sqrt(w**2))
        return dKxx

    @staticmethod
    @lazyjit('f8[:,:,:,:](f8[:], f8[:], f8, f8)', warn=False)
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

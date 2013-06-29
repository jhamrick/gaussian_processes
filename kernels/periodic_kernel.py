import numpy as np
from numpy import exp, sin, cos

import sympy as sym

from base_kernel import BaseKernel, lazyjit


class PeriodicKernel(object):
    """Represents a periodic kernel function, of the form:

    $$k(x_1, x_2) = h^2\exp(-\frac{2\sin^2(\frac{x_1-x_2}{2p})}{w^2})$$

    References
    ----------
    Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian processes
        for machine learning. MIT Press.

    """

    __metaclass__ = BaseKernel
    __slots__ = ['h', 'w', 'p']

    _sym_h = sym.Symbol('h')
    _sym_w = sym.Symbol('w')
    _sym_p = sym.Symbol('p')
    _sym_d = sym.Symbol('d')

    def __init__(self, h, w, p):
        """Create a PeriodicKernel object at specific parameter values.

        Parameters
        ----------
        h : number
            Output scale kernel parameter
        w : number
            Input scale (Gaussian standard deviation) kernel parameter
        p : number
            Period kernel parameter

        """

        self.h = h
        self.w = w
        self.p = p

    @property
    def sym_K(self):
        h = self._sym_h
        w = self._sym_w
        p = self._sym_p
        d = self._sym_d

        h2 = h ** 2
        w2 = w ** 2

        f = h2 * sym.exp(-2.*(sym.sin(d / (2.*p)) ** 2) / w2)
        return f

    def copy(self):
        return PeriodicKernel(*self.params)

    @property
    def params(self):
        return (self.h, self.w, self.p)

    @params.setter
    def params(self, val):
        self.h, self.w, self.p = val

    @staticmethod
    @lazyjit('f8[:,:](f8[:], f8[:], f8, f8, f8)', warn=False)
    def _K(x1, x2, h, w, p):
        Kxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                Kxx[i, j] = h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)
        return Kxx

    @staticmethod
    @lazyjit('f8[:,:](f8[:], f8[:], f8, f8, f8)', warn=False)
    def _dK_dh(x1, x2, h, w, p):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = 2.0*h*exp(-2.0*sin(0.5*d/p)**2/w**2)
        return dKxx

    @staticmethod
    @lazyjit('f8[:,:](f8[:], f8[:], f8, f8, f8)', warn=False)
    def _dK_dw(x1, x2, h, w, p):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = 4.0*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2/w**3
        return dKxx

    @staticmethod
    @lazyjit('f8[:,:](f8[:], f8[:], f8, f8, f8)', warn=False)
    def _dK_dp(x1, x2, h, w, p):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = 2.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**2*w**2)
        return dKxx

    @staticmethod
    @lazyjit('f8[:,:,:](f8[:], f8[:], f8, f8, f8)', warn=False)
    def _jacobian(x1, x2, h, w, p):
        dKxx = np.empty((3, x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[0, i, j] = 2.0*h*exp(-2.0*sin(0.5*d/p)**2/w**2)
                dKxx[1, i, j] = 4.0*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2/w**3
                dKxx[2, i, j] = 2.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**2*w**2)
        return dKxx

    @staticmethod
    @lazyjit('f8[:,:](f8[:], f8[:], f8, f8, f8)', warn=False)
    def _d2K_dhdh(x1, x2, h, w, p):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = 2.0*exp(-2.0*sin(0.5*d/p)**2/w**2)
        return dKxx

    @staticmethod
    @lazyjit('f8[:,:](f8[:], f8[:], f8, f8, f8)', warn=False)
    def _d2K_dhdw(x1, x2, h, w, p):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = 8.0*h*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2/w**3
        return dKxx

    @staticmethod
    @lazyjit('f8[:,:](f8[:], f8[:], f8, f8, f8)', warn=False)
    def _d2K_dhdp(x1, x2, h, w, p):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = 4.0*d*h*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**2*w**2)
        return dKxx

    @staticmethod
    @lazyjit('f8[:,:](f8[:], f8[:], f8, f8, f8)', warn=False)
    def _d2K_dwdh(x1, x2, h, w, p):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = 8.0*h*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2/w**3
        return dKxx

    @staticmethod
    @lazyjit('f8[:,:](f8[:], f8[:], f8, f8, f8)', warn=False)
    def _d2K_dwdw(x1, x2, h, w, p):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = -12.0*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2/w**4 + 16.0*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**4/w**6
        return dKxx

    @staticmethod
    @lazyjit('f8[:,:](f8[:], f8[:], f8, f8, f8)', warn=False)
    def _d2K_dwdp(x1, x2, h, w, p):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = -4.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**2*w**3) + 8.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**3*cos(0.5*d/p)/(p**2*w**5)
        return dKxx

    @staticmethod
    @lazyjit('f8[:,:](f8[:], f8[:], f8, f8, f8)', warn=False)
    def _d2K_dpdh(x1, x2, h, w, p):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = 4.0*d*h*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**2*w**2)
        return dKxx

    @staticmethod
    @lazyjit('f8[:,:](f8[:], f8[:], f8, f8, f8)', warn=False)
    def _d2K_dpdw(x1, x2, h, w, p):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = -4.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**2*w**3) + 8.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**3*cos(0.5*d/p)/(p**2*w**5)
        return dKxx

    @staticmethod
    @lazyjit('f8[:,:](f8[:], f8[:], f8, f8, f8)', warn=False)
    def _d2K_dpdp(x1, x2, h, w, p):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = d**2*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2/(p**4*w**2) - 1.0*d**2*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*cos(0.5*d/p)**2/(p**4*w**2) + 4.0*d**2*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2*cos(0.5*d/p)**2/(p**4*w**4) - 4.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**3*w**2)
        return dKxx

    @staticmethod
    @lazyjit('f8[:,:,:,:](f8[:], f8[:], f8, f8, f8)', warn=False)
    def _hessian(x1, x2, h, w, p):
        dKxx = np.empty((3, 3, x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                if d == 0:
                    dd = 1
                else:
                    dd = 0

                # h
                dKxx[0, 0, i, j] = 2.0*exp(-2.0*sin(0.5*d/p)**2/w**2)
                dKxx[0, 1, i, j] = 8.0*h*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2/w**3
                dKxx[0, 2, i, j] = 4.0*d*h*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**2*w**2)

                # w
                dKxx[1, 0, i, j] = 8.0*h*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2/w**3
                dKxx[1, 1, i, j] = -12.0*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2/w**4 + 16.0*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**4/w**6
                dKxx[1, 2, i, j] = -4.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**2*w**3) + 8.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**3*cos(0.5*d/p)/(p**2*w**5)

                # p
                dKxx[2, 0, i, j] = 4.0*d*h*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**2*w**2)
                dKxx[2, 1, i, j] = -4.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**2*w**3) + 8.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**3*cos(0.5*d/p)/(p**2*w**5)
                dKxx[2, 2, i, j]= d**2*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2/(p**4*w**2) - 1.0*d**2*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*cos(0.5*d/p)**2/(p**4*w**2) + 4.0*d**2*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2*cos(0.5*d/p)**2/(p**4*w**4) - 4.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**3*w**2)

        return dKxx

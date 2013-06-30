__all__ = ['PeriodicKernel']

import numpy as np
import sympy as sym

from numpy import exp, sin, cos
from functools import wraps
from . import Kernel

from util import staticlazyjit
staticlazyjit_mat = staticlazyjit(
    'f8[:,:](f8[:], f8[:], f8, f8, f8)', warn=False)


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
        self.h = h #: Output scale kernel parameter
        self.w = w #: Input scale kernel parameter
        self.p = p #: Period kernel parameter

    @property
    def params(self):
        r"""
        Kernel parameters.

        Returns
        -------
        params : tuple ``(h, w, p)``

        """
        return (self.h, self.w, self.p)

    @params.setter
    def params(self, val):
        self.h, self.w, self.p = val

    @property
    @wraps(Kernel.sym_K)
    def sym_K(self):
        h = sym.Symbol('h')
        w = sym.Symbol('w')
        p = sym.Symbol('p')
        d = sym.Symbol('d')

        h2 = h ** 2
        w2 = w ** 2

        f = h2 * sym.exp(-2.*(sym.sin(d / (2.*p)) ** 2) / w2)
        return f

    @staticlazyjit_mat
    @wraps(Kernel._K)
    def _K(x1, x2, h, w, p):
        Kxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                Kxx[i, j] = h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)
        return Kxx

    @staticlazyjit('f8[:,:,:](f8[:], f8[:], f8, f8, f8)', warn=False)
    @wraps(Kernel._jacobian)
    def _jacobian(x1, x2, h, w, p):
        dKxx = np.empty((3, x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[0, i, j] = 2.0*h*exp(-2.0*sin(0.5*d/p)**2/w**2)
                dKxx[1, i, j] = 4.0*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2/w**3
                dKxx[2, i, j] = 2.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**2*w**2)
        return dKxx

    @staticlazyjit('f8[:,:,:,:](f8[:], f8[:], f8, f8, f8)', warn=False)
    @wraps(Kernel._hessian)
    def _hessian(x1, x2, h, w, p):
        dKxx = np.empty((3, 3, x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
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

    @staticlazyjit_mat
    def _dK_dh(x1, x2, h, w, p):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = 2.0*h*exp(-2.0*sin(0.5*d/p)**2/w**2)
        return dKxx

    @staticlazyjit_mat
    def _dK_dw(x1, x2, h, w, p):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = 4.0*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2/w**3
        return dKxx

    @staticlazyjit_mat
    def _dK_dp(x1, x2, h, w, p):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = 2.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**2*w**2)
        return dKxx

    @staticlazyjit_mat
    def _d2K_dhdh(x1, x2, h, w, p):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = 2.0*exp(-2.0*sin(0.5*d/p)**2/w**2)
        return dKxx

    @staticlazyjit_mat
    def _d2K_dhdw(x1, x2, h, w, p):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = 8.0*h*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2/w**3
        return dKxx

    @staticlazyjit_mat
    def _d2K_dhdp(x1, x2, h, w, p):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = 4.0*d*h*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**2*w**2)
        return dKxx

    @staticlazyjit_mat
    def _d2K_dwdh(x1, x2, h, w, p):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = 8.0*h*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2/w**3
        return dKxx

    @staticlazyjit_mat
    def _d2K_dwdw(x1, x2, h, w, p):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = -12.0*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2/w**4 + 16.0*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**4/w**6
        return dKxx

    @staticlazyjit_mat
    def _d2K_dwdp(x1, x2, h, w, p):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = -4.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**2*w**3) + 8.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**3*cos(0.5*d/p)/(p**2*w**5)
        return dKxx

    @staticlazyjit_mat
    def _d2K_dpdh(x1, x2, h, w, p):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = 4.0*d*h*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**2*w**2)
        return dKxx

    @staticlazyjit_mat
    def _d2K_dpdw(x1, x2, h, w, p):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = -4.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**2*w**3) + 8.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**3*cos(0.5*d/p)/(p**2*w**5)
        return dKxx

    @staticlazyjit_mat
    def _d2K_dpdp(x1, x2, h, w, p):
        dKxx = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                d = x1[i] - x2[j]
                dKxx[i, j] = d**2*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2/(p**4*w**2) - 1.0*d**2*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*cos(0.5*d/p)**2/(p**4*w**2) + 4.0*d**2*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)**2*cos(0.5*d/p)**2/(p**4*w**4) - 4.0*d*h**2*exp(-2.0*sin(0.5*d/p)**2/w**2)*sin(0.5*d/p)*cos(0.5*d/p)/(p**3*w**2)
        return dKxx

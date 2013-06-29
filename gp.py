__all__ = ["GP"]

import numpy as np
import copy


def memoprop(f):
    """Memoized property.

    When the property is accessed for the first time, the return value
    is stored and that value is given on subsequent calls. The memoized
    value can be cleared by calling 'del prop', where prop is the name
    of the property.

    """
    fname = f.__name__

    def fget(self):
        if fname not in self._memoized:
            self._memoized[fname] = f(self)
        return self._memoized[fname]

    def fdel(self):
        del self._memoized[fname]

    prop = property(fget=fget, fdel=fdel, doc=f.__doc__)
    return prop


class GP(object):
    """Gaussian Process object.

    References
    ----------
    Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian processes
        for machine learning. MIT Press.

    """

    def __init__(self, K, x, y, s=0):
        """Initialize the GP.

        Parameters
        ----------
        K : function
            Kernel function, which takes two vectors as input and returns
            their inner product.
        x : numpy.ndarray
            Vector of input points
        y : numpy.ndarray
            Vector of input observations
        s : number (default=0)
            Observation noise parameter

        """
        self._memoized = {}
        self.K = K
        self.x = x
        self.y = y
        self.s = s

    def copy(self):
        new_gp = GP(self.K.copy(), self.x, self.y, s=self.s)
        new_gp._memoized = copy.deepcopy(self._memoized)
        return new_gp

    @property
    def x(self):
        """Vector of input points."""
        return self._x

    @x.setter
    def x(self, val):
        assert val.ndim == 2, val.ndim
        self._memoized = {}
        self._x = val.copy()

    @property
    def y(self):
        """Vector of input observations."""
        return self._y

    @y.setter
    def y(self, val):
        assert val.ndim == 2, val.ndim
        self._memoized = {}
        self._y = val.copy()

    @property
    def s(self):
        """Observation noise parameter."""
        return self._s

    @s.setter
    def s(self, val):
        self._memoized = {}
        self._s = val

    @property
    def params(self):
        """Kernel parameters."""
        return tuple(list(self.K.params) + [self._s])

    @params.setter
    def params(self, val):
        params = self.params
        if params[:-1] != val[:-1]:
            self._memoized = {}
            self.K.params = val[:-1]
        if params[-1] != val[-1]:
            self.s = val[-1]

    @memoprop
    def Kxx(self):
        """The kernel covariance matrix:

        $$K_{xx} = K(x, x') + s^2\delta(x-x'),$$

        where $\delta$ is the Dirac delta function.

        """
        x, s = self._x, self._s
        K = self.K(x, x)
        K += np.eye(x.size) * (s ** 2)
        if np.isnan(K).any():
            print self.K.params
            raise ArithmeticError("Kxx contains invalid values")
        return K

    @memoprop
    def Lxx(self):
        """Cholesky decomposition of K(x, x')"""
        return np.linalg.cholesky(self.Kxx)

    @memoprop
    def inv_Lxx(self):
        """Inverse cholesky decomposition of K(x, x')"""
        return np.linalg.inv(self.Lxx)

    @memoprop
    def inv_Kxx(self):
        """Inverse of K(x, x')"""
        Li = self.inv_Lxx
        return np.dot(Li.T, Li)

    @memoprop
    def inv_Kxx_y(self):
        """Dot product of inv(K(x, x')) and y"""
        return np.dot(self.inv_Kxx, self._y)

    @memoprop
    def log_lh(self):
        """The log likelihood of y given x and theta.

        This is computing Eq. 5.8 of Rasmussen & Williams (2006):

        $$\log{p(\mathbf{y} | X \mathbf{\theta})} = -\frac{1}{2}\mathbf{y}^\top K_y^{-1}\mathbf{y} - \frac{1}{2}\log{|K_y|}-\frac{n}{2}\log{2\pi}$$

        """
        y, K = self._y, self.Kxx
        sign, logdet = np.linalg.slogdet(K)
        if sign != 1:
            return -np.inf
        try:
            Kiy = self.inv_Kxx_y
        except np.linalg.LinAlgError:
            return -np.inf

        data_fit = -0.5 * np.dot(y.T, Kiy)
        complexity_penalty = -0.5 * logdet
        constant = -0.5 * y.size * np.log(2 * np.pi)
        llh = data_fit + complexity_penalty + constant
        return llh

    @property
    def lh(self):
        """The likelihood of y given x and theta. See GP.log_lh"""
        return np.exp(self.log_lh)

    @memoprop
    def dloglh_dtheta(self):
        """The partial derivatives of the marginal log likelihood with
        respect to its parameters `\theta`.

        See Eq. 5.9 of Rasmussen & Williams (2006).

        References
        ----------
        Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian processes
            for machine learning. MIT Press.

        """

        x, y = self._x, self._y
        try:
            Ki = self.inv_Kxx
        except np.linalg.LinAlgError:
            return np.array([-np.inf for p in self.params])

        nparam = len(self.params)

        # compute kernel jacobian
        dK_dtheta = np.empty((nparam, y.size, y.size))
        dK_dtheta[:-1] = self.K.jacobian(x, x)
        dK_dtheta[-1] = np.eye(y.size) * 2 * self._s

        dloglh = np.empty(nparam)
        for i in xrange(dloglh.size):
            k = np.dot(Ki, dK_dtheta[i])
            t0 = 0.5 * np.dot(y.T, np.dot(k, self.inv_Kxx_y))
            t1 = -0.5 * np.trace(k)
            dloglh[i] = t0 + t1

        return dloglh

    @memoprop
    def dlh_dtheta(self):
        """The partial derivatives of the marginal likelihood with
        respect to its parameters `\theta`.

        See Eq. 5.9 of Rasmussen & Williams (2006).

        References
        ----------
        Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian processes
            for machine learning. MIT Press.

        """

        x, y, K, Ki = self._x, self._y, self.Kxx, self.inv_Kxx
        Kiy = self.inv_Kxx_y
        nparam = len(self.params)

        dK_dtheta = np.empty((nparam, y.size, y.size))
        dK_dtheta[:-1] = self.K.jacobian(x, x)
        dK_dtheta[-1] = np.eye(y.size) * 2 * self._s

        lh = self.lh
        dlh = np.empty(nparam)
        for i in xrange(dlh.size):
            KidK = np.dot(Ki, dK_dtheta[i])
            t0 = np.dot(y.T, np.dot(KidK, Kiy))
            t1 = np.trace(KidK)
            dlh[i] = 0.5 * lh * (t0 - t1)

        return dlh

    @memoprop
    def d2lh_dtheta2(self):
        """The second partial derivatives of the marginal likelihood
        with respect to its parameters `\theta`.

        """

        y, x, K, Ki = self._y, self._x, self.Kxx, self.inv_Kxx
        Kiy = self.inv_Kxx_y
        nparam = len(self.params)

        # first kernel derivatives
        dK = np.empty((nparam, y.size, y.size))
        dK[:-1] = self.K.jacobian(x, x)
        dK[-1] = np.eye(y.size) * 2 * self._s
        dKi = [np.dot(-Ki, np.dot(dK[i], Ki)) for i in xrange(nparam)]

        # second kernel derivatives
        d2K = np.zeros((nparam, nparam, y.size, y.size))
        d2K[:-1, :-1] = self.K.hessian(x, x)
        d2K[-1, -1] = np.eye(y.size) * 2

        # likelihood
        lh = self.lh
        # first derivative of the likelihood
        dlh = self.dlh_dtheta

        d2lh = np.empty((nparam, nparam))
        for i in xrange(nparam):
            KidK_i = np.dot(Ki, dK[i])
            ydKi_iy = np.dot(y.T, np.dot(KidK_i, Kiy))
            ydKi_iy_tr = ydKi_iy - np.trace(KidK_i)

            for j in xrange(nparam):
                dKi_jdK_i = np.dot(dKi[j], dK[i])
                d_ydKi_iy = (
                    np.dot(y.T, np.dot(dKi_jdK_i, Kiy)) +
                    np.dot(Kiy.T, np.dot(d2K[i, j], Kiy)) +
                    np.dot(Kiy.T, np.dot(dK[i], np.dot(dKi[j], y))))
                d_tr = np.trace(dKi_jdK_i + np.dot(Ki, d2K[i, j]))

                t0 = dlh[j] * ydKi_iy_tr
                t1 = lh * (d_ydKi_iy - d_tr)
                d2lh[i, j] = 0.5 * (t0 + t1)

        return d2lh

    def Kxoxo(self, xo):
        """Kernel covariance matrix of new sample points xo:

        K(xo, xo')

        """
        return self.K(xo, xo)

    def Kxxo(self, xo):
        """Kernel covariance matrix/vector between given points x and
        new points xo:

        K(x, xo)

        """
        return self.K(self._x, xo)

    def Kxox(self, xo):
        """Kernel covariance matrix/vector between given points x and
        new points xo:

        K(xo, x)

        """
        return self.K(xo, self._x)

    def mean(self, xo):
        """Predictive mean of the GP.

        This is computing Eq. 2.23 of Rasmussen & Williams (2006):

        $$\mathbf{m}=K(X_*, X)[K(X, X) + \sigma_n^2]^{-1}\mathbf{y}$$

        References
        ----------
        Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian processes
            for machine learning. MIT Press.

        """
        return np.dot(self.Kxox(xo), self.inv_Kxx_y)

    def cov(self, xo):
        """Predictive covariance of the GP.

        This is computing Eq. 2.24 of Rasmussen & Williams (2006):

        $$\mathbf{C}=K(X_*, X_*) - K(X_*, X)[K(X, X) + \sigma_n^2]^{-1}K(X, X_*)$$

        References
        ----------
        Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian processes
            for machine learning. MIT Press.

        """
        Kxoxo = self.Kxoxo(xo)
        Kxox = self.Kxox(xo)
        Kxxo = self.Kxxo(xo)
        return Kxoxo - np.dot(Kxox, np.dot(self.inv_Kxx, Kxxo))

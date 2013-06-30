__all__ = ["GP"]

import numpy as np
import copy


def memoprop(f):
    """
    Memoized property.

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
    r"""
    Gaussian Process object.

    Parameters
    ----------
    K : function
        Kernel function, which takes two vectors as input and returns
        their inner product.
    x : numpy.ndarray
        :math:`n\times d` array of input locations
    y : numpy.ndarray
        :math:`n\times 1` array of input observations
    s : number (default=0)
        Observation noise parameter

    References
    ----------
    Rasmussen, C. E., & Williams, C. K. I. (2006). `Gaussian processes
    for machine learning.` MIT Press.

    """

    def __init__(self, K, x, y, s=0):
        r"""
        Initialize the GP.

        Parameters
        ----------
        K : function
            Kernel function, which takes two vectors as input and returns
            their inner product.
        x : numpy.ndarray
            :math:`n\times d` array of input locations
        y : numpy.ndarray
            :math:`n\times 1` array of input observations
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
        r"""
        Vector of input locations.

        Returns
        -------
        x : numpy.ndarray
            :math:`n\times d` array, where :math:`n` is the number of
            locations and :math:`d` is the number of dimensions.

        """
        return self._x

    @x.setter
    def x(self, val):
        assert val.ndim == 2, val.ndim
        self._memoized = {}
        self._x = val.copy()

    @property
    def y(self):
        r"""
        Vector of input observations.

        Returns
        -------
        y : numpy.ndarray
            :math:`n\times 1` array, where :math:`n` is the number of
            observations.

        """
        return self._y

    @y.setter
    def y(self, val):
        assert val.ndim == 2, val.ndim
        self._memoized = {}
        self._y = val.copy()

    @property
    def s(self):
        r"""
        Standard deviation of the observation noise for the gaussian
        process.

        Returns
        -------
        s : float

        """
        return self._s

    @s.setter
    def s(self, val):
        self._memoized = {}
        self._s = val

    @property
    def params(self):
        r"""
        Gaussian process parameters.

        Returns
        -------
        params : tuple
           Consists of the kernel's parameters, `self.K.params`, and the
           observation noise parameter, :math:`s`, in that order.

        """
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
        r"""
        Kernel covariance matrix :math:`\mathbf{K}_{xx}`, where the
        entry at index :math:`(i, j)` is defined as:

        .. math:: K_{x_ix_j} = K(x_i, x_j) + s^2\delta(x_i-x_j),

        where :math:`K(\cdot{})` is the kernel function, :math:`s` is the
        standard deviation of the observation noise, and :math:`\delta`
        is the Dirac delta function.

        Returns
        -------
        Kxx : numpy.ndarray
            :math:`n\times n` covariance matrix

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
        r"""
        Cholesky decomposition of the kernel covariance matrix. The
        value is :math:`\mathbf{L}_{xx}`, such that
        :math:`\mathbf{K}_{xx} = \mathbf{L}_{xx}\mathbf{L}_{xx}^\top`.

        Returns
        -------
        Lxx : numpy.ndarray
            :math:`n\times n` lower triangular matrix

        """
        return np.linalg.cholesky(self.Kxx)

    @memoprop
    def inv_Lxx(self):
        r"""
        Inverse cholesky decomposition of the kernel covariance
        matrix. The value is :math:`\mathbf{L}_{xx}^{-1}`, such that:

        .. math:: \mathbf{K}_{xx} = \mathbf{L}_{xx}\mathbf{L}_{xx}^\top

        Returns
        -------
        inv_Lxx : numpy.ndarray
            :math:`n\times n` matrix

        """
        return np.linalg.inv(self.Lxx)

    @memoprop
    def inv_Kxx(self):
        r"""
        Inverse kernel covariance matrix, :math:`\mathbf{K}_{xx}^{-1}`.

        Returns
        -------
        inv_Kxx : numpy.ndarray
            :math:`n\times n` matrix

        """
        Li = self.inv_Lxx
        return np.dot(Li.T, Li)

    @memoprop
    def inv_Kxx_y(self):
        r"""
        Dot product of the inverse kernel covariance matrix and vector
        of observations, defined as
        :math:`\mathbf{K}_{xx}^{-1}\mathbf{y}`.

        Returns
        -------
        inv_Kxx_y : numpy.ndarray
            :math:`n\times 1` array

        """
        return np.dot(self.inv_Kxx, self._y)

    @memoprop
    def log_lh(self):
        r"""
        Marginal log likelihood of observations :math:`\mathbf{y}` given
        locations :math:`\mathbf{x}` and kernel parameters
        :math:`\theta`. It is defined by Eq. 5.8 of Rasmussen & Williams
        (2006):

        .. math::

            \log{p(\mathbf{y} | \mathbf{x}, \mathbf{\theta})} = -\frac{1}{2}\mathbf{y}^\top \mathbf{K}_{xx}^{-1}\mathbf{y} - \frac{1}{2}\log{\left|\mathbf{K}_{xx}\right|}-\frac{d}{2}\log{2\pi},

        where :math:`d` is the dimensionality of :math:`\mathbf{x}`.

        Returns
        -------
        log_lh : float
            Marginal log likelihood

        References
        ----------
        Rasmussen, C. E., & Williams, C. K. I. (2006). `Gaussian processes
        for machine learning.` MIT Press.

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

    @memoprop
    def lh(self):
        r"""
        Marginal likelihood of observations :math:`\mathbf{y}` given
        locations :math:`\mathbf{x}` and kernel parameters
        :math:`\theta`. It is defined as:

        .. math::

            p(\mathbf{y} | \mathbf{x}, \mathbf{\theta}) = \left(2\pi\right)^{-\frac{d}{2}}\left|\mathbf{K}_{xx}\right|^{-\frac{1}{2}}\exp\left(-\frac{1}{2}\mathbf{y}^\top\mathbf{K}_{xx}^{-1}\mathbf{y}\right)

        where :math:`d` is the dimensionality of :math:`\mathbf{x}`.

        Returns
        -------
        lh : float
            Marginal likelihood

        """
        return np.exp(self.log_lh)

    @memoprop
    def dloglh_dtheta(self):
        r"""
        Vector of first partial derivatives of the marginal log
        likelihood with respect to its parameters :math:`\theta`. It is
        defined by Equation 5.9 of Rasmussen & Williams (2006):

        .. math::

            \frac{\partial}{\partial\theta_i}\log{p(\mathbf{y}|\mathbf{x},\theta)}=\frac{1}{2}\mathbf{y}^\top\mathbf{K}_{xx}^{-1}\frac{\partial\mathbf{K}_{xx}}{\partial\theta_i}\mathbf{K}_{xx}^{-1}\mathbf{y}-\frac{1}{2}\mathbf{tr}\left(\mathbf{K}_{xx}^{-1}\frac{\partial\mathbf{K}_{xx}}{\partial\theta_i}\right)

        Returns
        -------
        dloglh_dtheta : numpy.ndarray
            :math:`n_\theta`-length vector of derivatives, where
            :math:`n_\theta` is the number of parameters (equivalent to
            ``len(self.params)``).

        References
        ----------
        Rasmussen, C. E., & Williams, C. K. I. (2006). `Gaussian processes
        for machine learning.` MIT Press.

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
        r"""
        Vector of first partial derivatives of the marginal likelihood
        with respect to its parameters :math:`\theta`.

        Returns
        -------
        dlh_dtheta : numpy.ndarray
            :math:`n_\theta`-length vector of derivatives, where
            :math:`n_\theta` is the number of parameters (equivalent to
            ``len(self.params)``).

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
        r"""
        Matrix of second partial derivatives of the marginal likelihood
        with respect to its parameters :math:`\theta`.

        Returns
        -------
        d2lh_dtheta2 : numpy.ndarray
            :math:`n_\theta`-length vector of derivatives, where
            :math:`n_\theta` is the number of parameters (equivalent to
            ``len(self.params)``).

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
        r"""
        Kernel covariance matrix of new sample locations
        :math:`\mathbf{x^*}`, defined as :math:`K(\mathbf{x^*},
        \mathbf{x^*})`.

        Parameters
        ----------
        xo : numpy.ndarray
            :math:`m\times d` array of new sample locations

        Returns
        -------
        Kxoxo : numpy.ndarray
            :math:`m\times m` covariance matrix

        """
        return self.K(xo, xo)

    def Kxxo(self, xo):
        r"""
        Kernel covariance matrix between given locations
        :math:`\mathbf{x}` and new sample locations
        :math:`\mathbf{x^*}`, defined as
        :math:`K(\mathbf{x},\mathbf{x^*})`.

        Parameters
        ----------
        xo : numpy.ndarray
            :math:`m\times d` array of new sample locations

        Returns
        -------
        Kxxo : numpy.ndarray
            :math:`n\times m` covariance matrix

        """
        return self.K(self._x, xo)

    def Kxox(self, xo):
        r"""
        Kernel covariance matrix between new sample locations
        :math:`\mathbf{x^*}` and given locations :math:`\mathbf{x}`,
        defined as :math:`K(\mathbf{x^*},\mathbf{x})`.

        Parameters
        ----------
        xo : numpy.ndarray
            :math:`m\times d` array of new sample locations

        Returns
        -------
        Kxox : numpy.ndarray
            :math:`m\times n` covariance matrix

        """
        return self.K(xo, self._x)

    def mean(self, xo):
        r"""
        Predictive mean of the gaussian process, defined by Equation
        2.23 of Rasmussen & Williams (2006):

        .. math::

            \mathbf{m}(\mathbf{x^*})=K(\mathbf{x^*}, \mathbf{x})\mathbf{K}_{xx}^{-1}\mathbf{y}

        Parameters
        ----------
        xo : numpy.ndarray
            :math:`m\times d` array of new sample locations

        Returns
        -------
        mean : numpy.ndarray
            :math:`m\times d` array of predictive means

        References
        ----------
        Rasmussen, C. E., & Williams, C. K. I. (2006). `Gaussian processes
        for machine learning.` MIT Press.

        """
        return np.dot(self.Kxox(xo), self.inv_Kxx_y)

    def cov(self, xo):
        r"""
        Predictive covariance of the gaussian process, defined by
        Eq. 2.24 of Rasmussen & Williams (2006):

        .. math::

            \mathbf{C}=K(\mathbf{x^*}, \mathbf{x^*}) - K(\mathbf{x^*}, \mathbf{x})\mathbf{K}_{xx}^{-1}K(\mathbf{x}, \mathbf{x^*})

        Parameters
        ----------
        xo : numpy.ndarray
            :math:`m\times d` array of new sample locations

        Returns
        -------
        cov : numpy.ndarray
            :math:`m\times m` array of predictive covariances

        References
        ----------
        Rasmussen, C. E., & Williams, C. K. I. (2006). `Gaussian processes
        for machine learning.` MIT Press.

        """
        Kxoxo = self.Kxoxo(xo)
        Kxox = self.Kxox(xo)
        Kxxo = self.Kxxo(xo)
        return Kxoxo - np.dot(Kxox, np.dot(self.inv_Kxx, Kxxo))

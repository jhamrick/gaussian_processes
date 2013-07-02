__all__ = ["GP"]

import numpy as np
import copy
import matplotlib.pyplot as plt


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
    K : :class:`kernels.Kernel`
        Kernel object
    x : numpy.ndarray
        :math:`n\times d` array of input locations
    y : numpy.ndarray
        :math:`n\times 1` array of input observations
    s : number (default=0)
        Standard deviation of observation noise

    """

    def __init__(self, K, x, y, s=0):
        r"""
        Initialize the GP.

        """
        self._memoized = {}

        #: Kernel for the gaussian process, of type
        #: :class:`kernels.Kernel`
        self.K = K

        self.x = x
        self.y = y
        self.s = s

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

    def copy(self):
        """
        Create a copy of the gaussian process object.

        Returns
        -------
        gp : :class:`gp.GP`
            New gaussian process object

        """
        new_gp = GP(self.K.copy(), self.x, self.y, s=self.s)
        new_gp._memoized = copy.deepcopy(self._memoized)
        return new_gp

    @memoprop
    def Kxx(self):
        r"""
        Kernel covariance matrix :math:`\mathbf{K}_{xx}`.

        Returns
        -------
        Kxx : numpy.ndarray
            :math:`n\times n` covariance matrix

        Notes
        -----
        The entry at index :math:`(i, j)` is defined as:

        .. math:: K_{x_ix_j} = K(x_i, x_j) + s^2\delta(x_i-x_j),

        where :math:`K(\cdot{})` is the kernel function, :math:`s` is the
        standard deviation of the observation noise, and :math:`\delta`
        is the Dirac delta function.

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
        Cholesky decomposition of the kernel covariance matrix.

        Returns
        -------
        Lxx : numpy.ndarray
            :math:`n\times n` lower triangular matrix

        Notes
        -----
        The value is :math:`\mathbf{L}_{xx}`, such that
        :math:`\mathbf{K}_{xx} = \mathbf{L}_{xx}\mathbf{L}_{xx}^\top`.

        """
        return np.linalg.cholesky(self.Kxx)

    @memoprop
    def inv_Lxx(self):
        r"""
        Inverse cholesky decomposition of the kernel covariance matrix.

        Returns
        -------
        inv_Lxx : numpy.ndarray
            :math:`n\times n` matrix

        Notes
        -----
        The value is :math:`\mathbf{L}_{xx}^{-1}`, such that:

        .. math:: \mathbf{K}_{xx} = \mathbf{L}_{xx}\mathbf{L}_{xx}^\top

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
        Dot product of the inverse kernel covariance matrix and of
        observation vector.

        Returns
        -------
        inv_Kxx_y : numpy.ndarray
            :math:`n\times 1` array

        Notes
        -----
        This is defined as :math:`\mathbf{K}_{xx}^{-1}\mathbf{y}`.

        """
        return np.dot(self.inv_Kxx, self._y)

    @memoprop
    def log_lh(self):
        r"""
        Marginal log likelihood.

        Returns
        -------
        log_lh : float
            Marginal log likelihood

        Notes
        -----
        This is the log likelihood of observations :math:`\mathbf{y}`
        given locations :math:`\mathbf{x}` and kernel parameters
        :math:`\theta`. It is defined by Eq. 5.8 of [RW06]_:

        .. math::

            \log{p(\mathbf{y} | \mathbf{x}, \mathbf{\theta})} = -\frac{1}{2}\mathbf{y}^\top \mathbf{K}_{xx}^{-1}\mathbf{y} - \frac{1}{2}\log{\left|\mathbf{K}_{xx}\right|}-\frac{d}{2}\log{2\pi},

        where :math:`d` is the dimensionality of :math:`\mathbf{x}`.

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
        Marginal likelihood.

        Returns
        -------
        lh : float
            Marginal likelihood

        Notes
        -----
        This is the likelihood of observations :math:`\mathbf{y}` given
        locations :math:`\mathbf{x}` and kernel parameters
        :math:`\theta`. It is defined as:

        .. math::

            p(\mathbf{y} | \mathbf{x}, \mathbf{\theta}) = \left(2\pi\right)^{-\frac{d}{2}}\left|\mathbf{K}_{xx}\right|^{-\frac{1}{2}}\exp\left(-\frac{1}{2}\mathbf{y}^\top\mathbf{K}_{xx}^{-1}\mathbf{y}\right)

        where :math:`d` is the dimensionality of :math:`\mathbf{x}`.

        """
        return np.exp(self.log_lh)

    @memoprop
    def dloglh_dtheta(self):
        r"""
        Derivative of the marginal log likelihood.

        Returns
        -------
        dloglh_dtheta : numpy.ndarray
            :math:`n_\theta`-length vector of derivatives, where
            :math:`n_\theta` is the number of parameters (equivalent to
            ``len(self.params)``).

        Notes
        -----
        This is a vector of first partial derivatives of the log
        likelihood with respect to its parameters :math:`\theta`. It is
        defined by Equation 5.9 of [RW06]_:

        .. math::

            \frac{\partial}{\partial\theta_i}\log{p(\mathbf{y}|\mathbf{x},\theta)}=\frac{1}{2}\mathbf{y}^\top\mathbf{K}_{xx}^{-1}\frac{\partial\mathbf{K}_{xx}}{\partial\theta_i}\mathbf{K}_{xx}^{-1}\mathbf{y}-\frac{1}{2}\mathbf{tr}\left(\mathbf{K}_{xx}^{-1}\frac{\partial\mathbf{K}_{xx}}{\partial\theta_i}\right)

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
        Derivative of the marginal likelihood.

        Returns
        -------
        dlh_dtheta : numpy.ndarray
            :math:`n_\theta`-length vector of derivatives, where
            :math:`n_\theta` is the number of parameters (equivalent to
            ``len(self.params)``).

        Notes
        -----
        This is a vector of first partial derivatives of the likelihood
        with respect to its parameters :math:`\theta`.

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
        Second derivative of the marginal likelihood.

        Returns
        -------
        d2lh_dtheta2 : numpy.ndarray
            :math:`n_\theta`-length vector of derivatives, where
            :math:`n_\theta` is the number of parameters (equivalent to
            ``len(self.params)``).

        Notes
        -----
        This is a matrix of second partial derivatives of the likelihood
        with respect to its parameters :math:`\theta`.

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
        Kernel covariance matrix of new sample locations.

        Parameters
        ----------
        xo : numpy.ndarray
            :math:`m\times d` array of new sample locations

        Returns
        -------
        Kxoxo : numpy.ndarray
            :math:`m\times m` covariance matrix

        Notes
        -----
        This is defined as :math:`K(\mathbf{x^*}, \mathbf{x^*})`, where
        :math:`\mathbf{x^*}` are the new locations.

        """
        return self.K(xo, xo)

    def Kxxo(self, xo):
        r"""
        Kernel covariance matrix between given locations and new sample
        locations.

        Parameters
        ----------
        xo : numpy.ndarray
            :math:`m\times d` array of new sample locations

        Returns
        -------
        Kxxo : numpy.ndarray
            :math:`n\times m` covariance matrix

        Notes
        -----
        This is defined as :math:`K(\mathbf{x},\mathbf{x^*})`, where
        :math:`\mathbf{x}` are the given locations and
        :math:`\mathbf{x^*}` are the new sample locations.

        """
        return self.K(self._x, xo)

    def Kxox(self, xo):
        r"""
        Kernel covariance matrix between new sample locations and given
        locations.

        Parameters
        ----------
        xo : numpy.ndarray
            :math:`m\times d` array of new sample locations

        Returns
        -------
        Kxox : numpy.ndarray
            :math:`m\times n` covariance matrix

        Notes
        -----
        This is defined as :math:`K(\mathbf{x^*},\mathbf{x})`, where
        :math:`\mathbf{x^*}` are the new sample locations and
        :math:`\mathbf{x}` are the given locations

        """
        return self.K(xo, self._x)

    def mean(self, xo):
        r"""
        Predictive mean of the gaussian process.

        Parameters
        ----------
        xo : numpy.ndarray
            :math:`m\times d` array of new sample locations

        Returns
        -------
        mean : numpy.ndarray
            :math:`m\times d` array of predictive means

        Notes
        -----
        This is defined by Equation 2.23 of [RW06]_:

        .. math::

            \mathbf{m}(\mathbf{x^*})=K(\mathbf{x^*}, \mathbf{x})\mathbf{K}_{xx}^{-1}\mathbf{y}

        """
        return np.dot(self.Kxox(xo), self.inv_Kxx_y)

    def cov(self, xo):
        r"""
        Predictive covariance of the gaussian process.

        Parameters
        ----------
        xo : numpy.ndarray
            :math:`m\times d` array of new sample locations

        Returns
        -------
        cov : numpy.ndarray
            :math:`m\times m` array of predictive covariances

        Notes
        -----
        This is defined by Eq. 2.24 of [RW06]_:

        .. math::

            \mathbf{C}=K(\mathbf{x^*}, \mathbf{x^*}) - K(\mathbf{x^*}, \mathbf{x})\mathbf{K}_{xx}^{-1}K(\mathbf{x}, \mathbf{x^*})

        """
        Kxoxo = self.Kxoxo(xo)
        Kxox = self.Kxox(xo)
        Kxxo = self.Kxxo(xo)
        return Kxoxo - np.dot(Kxox, np.dot(self.inv_Kxx, Kxxo))

    def dm_dtheta(self, xo):
        r"""
        Derivative of the mean of the gaussian process with respect to
        its parameters, and evaluated at `xo`.

        Parameters
        ----------
        xo : numpy.ndarray
            :math:`m\times d` array of new sample locations

        Returns
        -------
        dm_dtheta : numpy.ndarray
            :math:`n_p\times m\times d` array, where :math:`n_p` is the
            number of parameters (see `params`).

        Notes
        -----
        The analytic form is:

        .. math::

            \frac{\partial}{\partial \theta_i}m(\mathbf{x^*})=\frac{\partial K(\mathbf{x^*}, \mathbf{x})}{\partial \theta_i}\mathbf{K}_{xx}^{-1}\mathbf{y} - K(\mathbf{x^*}, \mathbf{x})\mathbf{K}_{xx}^{-1}\frac{\partial \mathbf{K}_{xx}}{\partial \theta_i}\mathbf{K}_{xx}^{-1}\mathbf{y}

        """

        x, y, inv_Kxx = self._x, self._y, self.inv_Kxx
        Kxox = self.Kxox(xo)

        # dimensions
        n, d = x.shape
        m, d = xo.shape

        # compute kernel derivatives for s
        dKxox_ds = np.zeros((1, m, n))
        dKxx_ds = (np.eye(n) * 2 * self.s)[None]

        # all kernel partial derivatives
        dKxox_dtheta = np.concatenate([
            self.K.jacobian(xo, x), dKxox_ds], axis=0)
        dKxx_dtheta = np.concatenate([
            self.K.jacobian(x, x), dKxx_ds], axis=0)

        # number of parameters
        n_p = dKxx_dtheta.shape[0]

        dm = np.empty((n_p, m, 1))
        for i in xrange(n_p):
            inv_dKxx_dtheta = np.dot(inv_Kxx, np.dot(dKxx_dtheta[i], inv_Kxx))
            term1 = np.dot(dKxox_dtheta[i], np.dot(inv_Kxx, y))
            term2 = np.dot(Kxox, np.dot(inv_dKxx_dtheta, y))
            dm[i] = term1 - term2

        return dm

    def plot(self, ax=None, xlim=None, color='k', markercolor='r'):
        """
        Plot the predictive mean and variance of the gaussian process.

        Parameters
        ----------
        ax : `matplotlib.pyplot.axes.Axes` (optional)
            The axes on which to draw the graph. Defaults to
            ``plt.gca()`` if not given.
        xlim : (lower x limit, upper x limit) (optional)
            The limits of the x-axis. Defaults to the minimum and
            maximum of `x` if not given.
        color : str (optional)
            The line color to use. The default is 'k' (black).
        markercolor : str (optional)
            The marker color to use. The default is 'r' (red).

        """

        x, y = self._x, self._y
        n, d = x.shape
        if d != 1:
            raise ValueError("data has too many dimensions")
        x = x[:, 0]
        y = y[:, 0]

        if ax is None:
            ax = plt.gca()
        if xlim is None:
            xlim = (x.min(), x.max())

        X = np.linspace(xlim[0], xlim[1], 1000)
        mean = self.mean(X)[:, 0]
        cov = self.cov(X)
        std = np.sqrt(np.diag(cov))
        upper = mean + std
        lower = mean - std

        ax.fill_between(X, lower, upper, color=color, alpha=0.3)
        ax.plot(X, mean, lw=2, color=color)
        ax.plot(x, y, 'o', ms=7, color=markercolor)
        ax.set_xlim(*xlim)

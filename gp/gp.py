__all__ = ["GP"]

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optim
import logging

from scipy.linalg import cholesky, cho_solve
from copy import copy, deepcopy

from .ext import gp_c

logger = logging.getLogger("gp.gp")

DTYPE = np.float64
EPS = np.finfo(DTYPE).eps
MIN = np.log(np.exp2(DTYPE(np.finfo(DTYPE).minexp + 4)))


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
    K : :class:`~gp.kernels.base.Kernel`
        Kernel object
    x : numpy.ndarray
        :math:`n` array of input locations
    y : numpy.ndarray
        :math:`n` array of input observations
    s : number (default=0)
        Standard deviation of observation noise

    """

    def __init__(self, K, x, y, s=0):
        r"""
        Initialize the GP.

        """
        #: Kernel for the gaussian process, of type
        #: :class:`~gp.kernels.base.Kernel`
        self.K = K
        self._x = None
        self._y = None
        self._s = None
        self._memoized = {}

        self.x = x
        self.y = y
        self.s = s

    def __getstate__(self):
        state = {}
        state['K'] = self.K
        state['_x'] = self._x
        state['_y'] = self._y
        state['_s'] = self._s
        state['_memoized'] = self._memoized
        return state

    def __setstate__(self, state):
        self.K = state['K']
        self._x = state['_x']
        self._y = state['_y']
        self._s = state['_s']
        self._memoized = state['_memoized']

    def __copy__(self):
        state = self.__getstate__()
        cls = type(self)
        gp = cls.__new__(cls)
        gp.__setstate__(state)
        return gp

    def __deepcopy__(self, memo):
        state = deepcopy(self.__getstate__(), memo)
        cls = type(self)
        gp = cls.__new__(cls)
        gp.__setstate__(state)
        return gp

    def copy(self, deep=True):
        """
        Create a copy of the gaussian process object.

        Parameters
        ----------
        deep : bool (default=True)
            Whether to return a deep or shallow copy

        Returns
        -------
        gp : :class:`~gp.gp.GP`
            New gaussian process object

        """
        if deep:
            gp = deepcopy(self)
        else:
            gp = copy(self)
        return gp

    @property
    def x(self):
        r"""
        Vector of input locations.

        Returns
        -------
        x : numpy.ndarray
            :math:`n` array, where :math:`n` is the number of
            locations.

        """
        return self._x

    @x.setter
    def x(self, val):
        if np.any(val != self._x):
            self._memoized = {}
            self._x = np.array(val, copy=True, dtype=DTYPE)
            self._x.flags.writeable = False
        else: # pragma: no cover
            pass

    @property
    def y(self):
        r"""
        Vector of input observations.

        Returns
        -------
        y : numpy.ndarray
            :math:`n` array, where :math:`n` is the number of
            observations.

        """
        return self._y

    @y.setter
    def y(self, val):
        if np.any(val != self._y):
            self._memoized = {}
            self._y = np.array(val, copy=True, dtype=DTYPE)
            self._y.flags.writeable = False
            if self.y.shape != self.x.shape:
                raise ValueError("invalid shape for y: %s" % str(self.y.shape))
        else: # pragma: no cover
            pass

    @property
    def s(self):
        r"""
        Standard deviation of the observation noise for the gaussian
        process.

        Returns
        -------
        s : numpy.float64

        """
        return self._s

    @s.setter
    def s(self, val):
        if val < 0:
            raise ValueError("invalid value for s: %s" % val)

        if val != self._s:
            self._memoized = {}
            self._s = DTYPE(val)

    @property
    def params(self):
        r"""
        Gaussian process parameters.

        Returns
        -------
        params : numpy.ndarray
           Consists of the kernel's parameters, `self.K.params`, and the
           observation noise parameter, :math:`s`, in that order.

        """
        _params = np.empty(self.K.params.size + 1)
        _params[:-1] = self.K.params
        _params[-1] = self._s
        return _params

    @params.setter
    def params(self, val):
        if np.any(self.params != val):
            self._memoized = {}
            self.K.params = val[:-1]
            self.s = val[-1]
        else: # pragma: no cover
            pass

    def get_param(self, name):
        if name == 's':
            return self.s
        else:
            return getattr(self.K, name)

    def set_param(self, name, val):
        if name == 's':
            self.s = val
        else:
            p = getattr(self.K, name)
            if p != val:
                self._memoized = {}
                self.K.set_param(name, val)
            else: # pragma: no cover
                pass

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
        K += np.eye(x.size, dtype=DTYPE) * (s ** 2)
        return K

    @memoprop
    def Kxx_J(self):
        x = self._x
        return self.K.jacobian(x, x)

    @memoprop
    def Kxx_H(self):
        x = self._x
        return self.K.hessian(x, x)

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
        return cholesky(self.Kxx, lower=True, overwrite_a=False, check_finite=True)

    @memoprop
    def inv_Kxx(self):
        r"""
        Inverse kernel covariance matrix, :math:`\mathbf{K}_{xx}^{-1}`.

        Note that this inverse is provided mostly just for
        reference. If you actually need to use it, use the Cholesky
        decomposition (`self.Lxx`) instead.

        Returns
        -------
        inv_Kxx : numpy.ndarray
            :math:`n\times n` matrix

        """
        inv_Lxx = np.linalg.inv(self.Lxx)
        return np.dot(inv_Lxx.T, inv_Lxx)

    @memoprop
    def inv_Kxx_y(self):
        r"""
        Dot product of the inverse kernel covariance matrix and of
        observation vector.
        
        This uses scipy's cholesky solver to compute the solution.

        Returns
        -------
        inv_Kxx_y : numpy.ndarray
            :math:`n` array

        Notes
        -----
        This is defined as :math:`\mathbf{K}_{xx}^{-1}\mathbf{y}`.

        """
        inv_Kxx_y = cho_solve(
            (self.Lxx, True), self._y, 
            overwrite_b=False, check_finite=True)
        return inv_Kxx_y

    @memoprop
    def log_lh(self):
        r"""
        Marginal log likelihood.

        Returns
        -------
        log_lh : numpy.float64
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
        y = self._y
        K = self.Kxx
        try:
            Kiy = self.inv_Kxx_y
        except np.linalg.LinAlgError:
            return -np.inf

        return DTYPE(gp_c.log_lh(y, K, Kiy))

    @memoprop
    def lh(self):
        r"""
        Marginal likelihood.

        Returns
        -------
        lh : numpy.float64
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
        llh = self.log_lh
        if llh < MIN:
            return 0
        else:
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

        y = self._y
        dloglh = np.empty(len(self.params))
        try:
            Ki = self.inv_Kxx
        except np.linalg.LinAlgError:
            dloglh.fill(np.nan)
            return dloglh

        Kj = self.Kxx_J
        Kiy = self.inv_Kxx_y
        gp_c.dloglh_dtheta(y, Ki, Kj, Kiy, self._s, dloglh)
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

        y = self._y
        dlh = np.empty(len(self.params))
        try:
            Ki = self.inv_Kxx
        except np.linalg.LinAlgError:
            dlh.fill(np.nan)
            return dlh

        Kj = self.Kxx_J
        Kiy = self.inv_Kxx_y
        lh = self.lh
        gp_c.dlh_dtheta(y, Ki, Kj, Kiy, self._s, lh, dlh)
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

        y = self._y
        d2lh = np.empty((len(self.params), len(self.params)))
        try:
            Ki = self.inv_Kxx
        except np.linalg.LinAlgError:
            d2lh.fill(np.nan)
            return d2lh

        Kj = self.Kxx_J
        Kh = self.Kxx_H
        Kiy = self.inv_Kxx_y
        lh = self.lh
        dlh = self.dlh_dtheta

        gp_c.d2lh_dtheta2(y, Ki, Kj, Kh, Kiy, self._s, lh, dlh, d2lh)
        return d2lh

    def Kxoxo(self, xo):
        r"""
        Kernel covariance matrix of new sample locations.

        Parameters
        ----------
        xo : numpy.ndarray
            :math:`m` array of new sample locations

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
            :math:`m` array of new sample locations

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
            :math:`m` array of new sample locations

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
            :math:`m` array of new sample locations

        Returns
        -------
        mean : numpy.ndarray
            :math:`m` array of predictive means

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
            :math:`m` array of new sample locations

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
            :math:`m` array of new sample locations

        Returns
        -------
        dm_dtheta : numpy.ndarray
            :math:`n_p\times m` array, where :math:`n_p` is the
            number of parameters (see `params`).

        Notes
        -----
        The analytic form is:

        .. math::

            \frac{\partial}{\partial \theta_i}m(\mathbf{x^*})=\frac{\partial K(\mathbf{x^*}, \mathbf{x})}{\partial \theta_i}\mathbf{K}_{xx}^{-1}\mathbf{y} - K(\mathbf{x^*}, \mathbf{x})\mathbf{K}_{xx}^{-1}\frac{\partial \mathbf{K}_{xx}}{\partial \theta_i}\mathbf{K}_{xx}^{-1}\mathbf{y}

        """

        y = self._y
        Ki = self.inv_Kxx
        Kj = self.Kxx_J
        Kjxo = self.K.jacobian(xo, self._x)
        Kxox = self.Kxox(xo)

        dm = np.empty((len(self.params), xo.size))
        gp_c.dm_dtheta(y, Ki, Kj, Kjxo, Kxox, self._s, dm)

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

        if ax is None:
            ax = plt.gca()
        if xlim is None:
            xlim = (x.min(), x.max())

        X = np.linspace(xlim[0], xlim[1], 1000)
        mean = self.mean(X)
        cov = self.cov(X)
        std = np.sqrt(np.diag(cov))
        upper = mean + std
        lower = mean - std

        ax.fill_between(X, lower, upper, color=color, alpha=0.3)
        ax.plot(X, mean, lw=2, color=color)
        ax.plot(x, y, 'o', ms=5, color=markercolor)
        ax.set_xlim(*xlim)

__all__ = ['Kernel']

from functools import wraps
from copy import copy


class Kernel(object):

    def __getstate__(self):
        state = {'params': self.params}
        return state

    def __setstate__(self, state):
        self.params = state['params']

    def __copy__(self):
        return type(self)(*self.params)

    def __deepcopy__(self, memo):
        return type(self)(*self.params)

    def copy(self):
        """
        Create a copy of the kernel.

        Returns
        -------
        kernel : :class:`~gp.kernels.base.Kernel`
            New kernel function object of type ``type(self)``.

        """
        return copy(self)

    def sym_K(self):
        """
        Symbolic kernel function.

        Returns
        -------
        K : sympy.Expr
            A sympy expression representing the symbolic form of the
            kernel function.

        """
        raise NotImplementedError

    @property
    def params(self):
        """
        Kernel parameters.

        Returns
        -------
        params : tuple

        """
        raise NotImplementedError

    def K(self, x1, x2, out=None):
        r"""
        Kernel function evaluated at `x1` and `x2`.

        Parameters
        ----------
        x1 : numpy.ndarray with ``dtype='f8'``
            :math:`n`-length vector
        x2 : numpy.ndarray with ``dtype='f8'``
            :math:`m`-length vector

        Returns
        -------
        K : numpy.ndarray
            :math:`n\times m` array

        """
        raise NotImplementedError

    @wraps(K)
    def __call__(self, x1, x2, out=None):
        return self.K(x1, x2, out=out)

    def jacobian(self, x1, x2, out=None):
        r"""
        Jacobian of the kernel function evaluated at `x1` and `x2`.

        Parameters
        ----------
        x1 : numpy.ndarray with ``dtype='f8'``
            :math:`n`-length vector
        x2 : numpy.ndarray with ``dtype='f8'``
            :math:`m`-length vector

        Returns
        -------
        J : numpy.ndarray
            :math:`n_p\times n\times m` array, where :math:`n_p` is the
            number of kernel parameters. See `params`.

        """
        raise NotImplementedError

    def hessian(self, x1, x2, out=None):
        r"""
        Hessian of the kernel function evaluated at `x1` and `x2`.

        Parameters
        ----------
        x1 : numpy.ndarray with ``dtype='f8'``
            :math:`n`-length vector
        x2 : numpy.ndarray with ``dtype='f8'``
            :math:`m`-length vector

        Returns
        -------
        H : numpy.ndarray
            :math:`n_p\times n_p\times n\times m` array, where
            :math:`n_p` is the number of kernel parameters. See
            `params`.

        """
        raise NotImplementedError

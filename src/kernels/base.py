__all__ = ['Kernel']

from util import selfparams


class KernelMeta(type):

    def __new__(cls, name, parents, attrs):

        new_attrs = attrs.copy()

        for attr_name in attrs:

            # skip magic methods
            if attr_name.startswith("__"):
                continue

            # skip non-internal methods
            if not attr_name.startswith("_"):
                continue

            # get the function object
            attr = attrs[attr_name]
            if hasattr(attr, '__call__'):
                func = attr
            elif hasattr(attr, '__func__'):
                func = attr.__func__
            else:
                continue

            if not hasattr(func, '__name__'):
                continue

            # remove _ from beginning of name
            new_attr_name = attr_name[1:]
            new_attr = selfparams(func)

            new_attrs[new_attr_name] = new_attr

        if '__call__' not in attrs and 'K' in new_attrs:
            new_attrs['__call__'] = new_attrs['K']

        # create the class
        obj = super(KernelMeta, cls).__new__(
            cls, name, parents, new_attrs)
        return obj


class Kernel(object):

    __metaclass__ = KernelMeta

    def copy(self):
        """
        Create a copy of the kernel.

        Returns
        -------
        kernel : :class:`kernels.Kernel`
            New kernel function object of type ``type(self)``.

        """
        return type(self)(*self.params)

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

    @staticmethod
    def _K(x1, x2, *params):
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

    @staticmethod
    def _jacobian(x1, x2, *params):
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

    @staticmethod
    def _hessian(x1, x2, *params):
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

import numba


def selfparams(f):
    def wrapped_func(self, x1, x2):
        return f(x1.astype('f8', copy=False),
                 x2.astype('f8', copy=False),
                 *self.params)
    wrapped_func.__name__ = f.__name__
    wrapped_func.__doc__ = f.__doc__
    return wrapped_func


def lazyjit(*args, **kwargs):
    compiler = numba.jit(*args, **kwargs)

    def compile(f):
        f.compiled = None

        def thunk(*fargs, **fkwargs):
            if not f.compiled:
                f.compiled = compiler(f)
            return f.compiled(*fargs, **fkwargs)

        thunk.__name__ = f.__name__
        thunk.__doc__ = f.__doc__
        return thunk

    return compile

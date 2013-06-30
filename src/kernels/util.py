import numba
from functools import wraps


def selfparams(f):
    @wraps(f)
    def wrapper(self, x1, x2):
        return f(x1.astype('f8', copy=False),
                 x2.astype('f8', copy=False),
                 *self.params)
    if wrapper.__name__.startswith("_"):
        wrapper.__name__ = wrapper.__name__[1:]
    return wrapper


def lazyjit(*args, **kwargs):
    compiler = numba.jit(*args, **kwargs)

    def compile(f):
        f.compiled = None

        @wraps(f)
        def wrapper(*fargs, **fkwargs):
            if not f.compiled:
                f.compiled = compiler(f)
            return f.compiled(*fargs, **fkwargs)

        return wrapper

    return compile


def staticlazyjit(*args, **kwargs):
    lazyjitdeco = lazyjit(*args, **kwargs)

    def wrapper(f):
        f_jit = lazyjitdeco(f)
        f_static = staticmethod(f_jit)
        return f_static

    return wrapper

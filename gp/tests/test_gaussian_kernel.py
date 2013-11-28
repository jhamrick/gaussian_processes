import scipy.stats
import numpy as np
np.seterr(all='raise')
np.random.seed(2348)

from .. import GaussianKernel
from .util import opt, rand_params
from .test_kernels import check_params, check_jacobian, check_hessian

EPS = np.finfo(float).eps

N_BIG = opt['n_big_test_iters']
N_SMALL = opt['n_small_test_iters']
THRESH = opt['error_threshold']
DTHETA = opt['dtheta']


def make_random_kernel():
    h, w = rand_params('h', 'w')
    kernel = GaussianKernel(h, w)
    return kernel

######################################################################


def test_kernel_params():
    for i in xrange(N_BIG):
        params = rand_params('h', 'w')
        yield check_params, GaussianKernel, params


def test_K():
    x = np.linspace(-2, 2, 10)
    dx = x[:, None] - x[None, :]

    def check_K(kernel):
        K = kernel(x, x)
        h, w = kernel.params
        pdx = np.empty(dx.shape)
        for i, v in enumerate(dx.flat):
            try:
                pdx.flat[i] = scipy.stats.norm.pdf(v, loc=0, scale=w)
            except FloatingPointError:
                pdx.flat[i] = 0.0
        pdx *= h ** 2
        assert (abs(pdx - K) < THRESH).all()

    for i in xrange(N_BIG):
        kernel = make_random_kernel()
        yield check_K, kernel


def test_sym_K():
    x = np.linspace(-2, 2, 3)
    dx = x[:, None] - x[None, :]

    def check_sym_K(params):
        kernel = GaussianKernel(*params)
        K = kernel(x, x)
        sym_K = kernel.sym_K
        Ks = np.empty_like(K)
        for i in xrange(x.size):
            for j in xrange(x.size):
                Ks[i, j] = sym_K.evalf(subs={
                    'd': dx[i, j],
                    'h': params[0],
                    'w': params[1]
                })
        assert (abs(Ks - K) < THRESH).all()

    yield (check_sym_K, (1, 1))
    yield (check_sym_K, (1, 2))
    yield (check_sym_K, (2, 1))
    yield (check_sym_K, (0.5, 0.5))


def test_jacobian():
    x = np.linspace(-2, 2, 10)
    for i in xrange(N_SMALL):
        kernel = make_random_kernel()
        yield (check_jacobian, kernel, x)


def test_hessian():
    x = np.linspace(-2, 2, 10)
    for i in xrange(N_SMALL):
        kernel = make_random_kernel()
        yield (check_hessian, kernel, x)

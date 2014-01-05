import numpy as np
import pickle
from copy import copy, deepcopy

from .. import PeriodicKernel
from .util import opt, rand_params, seed, allclose
from .test_kernels import check_jacobian, check_hessian
from .test_kernels import check_params, check_invalid_params
from .test_kernels import check_dK_dtheta, check_d2K_dtheta2

seed()

EPS = opt['eps']
N_BIG = opt['n_big_test_iters']
N_SMALL = opt['n_small_test_iters']
DTHETA = opt['dtheta']


def make_random_kernel():
    h, w, p = rand_params('h', 'w', 'p')
    kernel = PeriodicKernel(h, w, p)
    return kernel

######################################################################


def test_kernel_params():
    seed()
    for i in xrange(N_BIG):
        params = rand_params('h', 'w', 'p')
        yield check_params, PeriodicKernel, params

    good_params = rand_params('h', 'w', 'p')
    bad_params = list(good_params)
    bad_params[0] = 0
    yield check_invalid_params, PeriodicKernel, good_params, bad_params

    bad_params = list(good_params)
    bad_params[1] = 0
    yield check_invalid_params, PeriodicKernel, good_params, bad_params

    bad_params = list(good_params)
    bad_params[2] = 0
    yield check_invalid_params, PeriodicKernel, good_params, bad_params


def test_K():
    seed()
    x = np.linspace(-2 * np.pi, 2 * np.pi, 16)
    dx = x[:, None] - x[None, :]

    def check_K(kernel):
        K1 = kernel(x, x)
        K2 = np.empty_like(K1)
        kernel(x, x, out=K2)
        h, w, p = kernel.params
        pdx = (h ** 2) * np.exp(-2. * (np.sin(dx / (2. * p)) ** 2) / (w ** 2))
        assert allclose(pdx, K1)
        assert allclose(pdx, K2)
        assert allclose(K1, K2)

    for i in xrange(N_BIG):
        kernel = make_random_kernel()
        yield check_K, kernel


def test_sym_K():
    x = np.linspace(-2, 2, 3)
    dx = x[:, None] - x[None, :]

    def check_sym_K(params):
        kernel = PeriodicKernel(*params)
        K = kernel(x, x)
        sym_K = kernel.sym_K
        Ks = np.empty_like(K)
        for i in xrange(x.size):
            for j in xrange(x.size):
                Ks[i, j] = sym_K.evalf(subs={
                    'd': dx[i, j],
                    'h': params[0],
                    'w': params[1],
                    'p': params[2]
                })

        assert allclose(Ks, K)

    yield (check_sym_K, (1, 1, 1))
    yield (check_sym_K, (1, 2, 2))
    yield (check_sym_K, (2, 1, 0.5))
    yield (check_sym_K, (0.5, 0.5, 1))


def test_jacobian():
    seed()
    x = np.linspace(-2 * np.pi, 2 * np.pi, 16)
    for i in xrange(N_SMALL):
        kernel = make_random_kernel()
        yield (check_jacobian, kernel, x)


def test_hessian():
    seed()
    x = np.linspace(-2 * np.pi, 2 * np.pi, 16)
    for i in xrange(N_SMALL):
        kernel = make_random_kernel()
        yield (check_hessian, kernel, x)


def test_dK_dh():
    seed()
    x = np.linspace(-2 * np.pi, 2 * np.pi, 16)
    for i in xrange(N_SMALL):
        kernel = make_random_kernel()
        yield (check_dK_dtheta, kernel, x, 'h', 0)


def test_dK_dw():
    seed()
    x = np.linspace(-2 * np.pi, 2 * np.pi, 16)
    for i in xrange(N_SMALL):
        kernel = make_random_kernel()
        yield (check_dK_dtheta, kernel, x, 'w', 1)


def test_dK_dp():
    seed()
    x = np.linspace(-2 * np.pi, 2 * np.pi, 16)
    for i in xrange(N_SMALL):
        kernel = make_random_kernel()
        yield (check_dK_dtheta, kernel, x, 'p', 2)


def test_d2K_dhdh():
    seed()
    x = np.linspace(-2 * np.pi, 2 * np.pi, 16)
    for i in xrange(N_SMALL):
        kernel = make_random_kernel()
        yield (check_d2K_dtheta2, kernel, x, 'h', 'h', 0)


def test_d2K_dhdw():
    seed()
    x = np.linspace(-2 * np.pi, 2 * np.pi, 16)
    for i in xrange(N_SMALL):
        kernel = make_random_kernel()
        yield (check_d2K_dtheta2, kernel, x, 'h', 'w', 1)


def test_d2K_dhdp():
    seed()
    x = np.linspace(-2 * np.pi, 2 * np.pi, 16)
    for i in xrange(N_SMALL):
        kernel = make_random_kernel()
        yield (check_d2K_dtheta2, kernel, x, 'h', 'p', 2)


def test_d2K_dwdh():
    seed()
    x = np.linspace(-2 * np.pi, 2 * np.pi, 16)
    for i in xrange(N_SMALL):
        kernel = make_random_kernel()
        yield (check_d2K_dtheta2, kernel, x, 'w', 'h', 0)


def test_d2K_dwdw():
    seed()
    x = np.linspace(-2 * np.pi, 2 * np.pi, 16)
    for i in xrange(N_SMALL):
        kernel = make_random_kernel()
        yield (check_d2K_dtheta2, kernel, x, 'w', 'w', 1)


def test_d2K_dwdp():
    seed()
    x = np.linspace(-2 * np.pi, 2 * np.pi, 16)
    for i in xrange(N_SMALL):
        kernel = make_random_kernel()
        yield (check_d2K_dtheta2, kernel, x, 'w', 'p', 2)


def test_d2K_dpdh():
    seed()
    x = np.linspace(-2 * np.pi, 2 * np.pi, 16)
    for i in xrange(N_SMALL):
        kernel = make_random_kernel()
        yield (check_d2K_dtheta2, kernel, x, 'p', 'h', 0)


def test_d2K_dpdw():
    seed()
    x = np.linspace(-2 * np.pi, 2 * np.pi, 16)
    for i in xrange(N_SMALL):
        kernel = make_random_kernel()
        yield (check_d2K_dtheta2, kernel, x, 'p', 'w', 1)


def test_d2K_dpdp():
    seed()
    x = np.linspace(-2 * np.pi, 2 * np.pi, 16)
    for i in xrange(N_SMALL):
        kernel = make_random_kernel()
        yield (check_d2K_dtheta2, kernel, x, 'p', 'p', 2)



def test_copy_method():
    kernel1 = make_random_kernel()
    kernel2 = kernel1.copy()

    assert kernel1.h == kernel2.h
    assert kernel1.w == kernel2.w
    assert kernel1.p == kernel2.p


def test_copy():
    kernel1 = make_random_kernel()
    kernel2 = copy(kernel1)

    assert kernel1.h == kernel2.h
    assert kernel1.w == kernel2.w
    assert kernel1.p == kernel2.p


def test_deepcopy():
    kernel1 = make_random_kernel()
    kernel2 = deepcopy(kernel1)

    assert kernel1.h == kernel2.h
    assert kernel1.w == kernel2.w
    assert kernel1.p == kernel2.p
    assert kernel1.h is not kernel2.h
    assert kernel1.w is not kernel2.w
    assert kernel1.p is not kernel2.p


def test_pickle():
    kernel1 = make_random_kernel()
    state = pickle.dumps(kernel1)
    kernel2 = pickle.loads(state)

    assert kernel1.h == kernel2.h
    assert kernel1.w == kernel2.w
    assert kernel1.p == kernel2.p
    assert kernel1.h is not kernel2.h
    assert kernel1.w is not kernel2.w
    assert kernel1.p is not kernel2.p

import pytest
import numpy as np
from numpy import dot
np.seterr(all='raise')

from .. import GP
from .. import GaussianKernel as kernel
from .util import opt, rand_params, seed, allclose
from .util import approx_deriv, make_xy, make_xo

EPS = opt['eps']
N_BIG = opt['n_big_test_iters']
N_SMALL = opt['n_small_test_iters']
DTHETA = opt['dtheta']
PFAIL = opt['pct_allowed_failures']
DTYPE = opt['dtype']

######################################################################


def make_gp():
    x, y = make_xy()
    gp = GP(kernel(1, 1), x, y, s=1)
    return gp


def make_random_gp():
    x, y = make_xy()
    h, w, s = rand_params('h', 'w', 's')
    gp = GP(kernel(h, w), x, y, s=s)
    return gp


def count_failures(check, n):
    seed()

    params = []
    failures = []
    for i in xrange(n):
        gp = make_random_gp()
        try:
            check(gp)
        except AssertionError as err:
            params.append(tuple(gp.params))
            failures.append(err.msg)

    pfail = 100 * len(failures) / n
    msg = "%s failed %d/%d (%.1f%%) times" % (
        check.__name__, len(failures), n, pfail)
    print zip(params, failures)
    assert pfail < PFAIL, msg

##################################################################


def test_mean():
    def check_mean(gp):
        gp.s = 0
        assert allclose(gp.mean(gp.x), gp.y)

    count_failures(check_mean, N_BIG)


def test_inv():
    def check_inv(gp):
        I = dot(gp.Kxx, gp.inv_Kxx)
        assert allclose(I, np.eye(I.shape[0]))

    count_failures(check_inv, N_SMALL)


def test_dloglh():
    def check_dloglh(gp):
        params = gp.params.copy()
        jac = gp.dloglh_dtheta

        approx_jac = np.empty(jac.shape)
        for i in xrange(len(params)):
            p0 = list(params)
            p0[i] -= DTHETA
            gp0 = gp.copy()
            gp0.params = p0

            p1 = list(params)
            p1[i] += DTHETA
            gp1 = gp.copy()
            gp1.params = p1

            approx_jac[i] = approx_deriv(
                gp0.log_lh, gp1.log_lh, DTHETA)

        assert allclose(jac, approx_jac)

    count_failures(check_dloglh, N_BIG)


def test_dlh():
    def check_dlh(gp):
        params = gp.params.copy()
        jac = gp.dlh_dtheta

        approx_jac = np.empty(jac.shape)
        for i in xrange(len(params)):
            p0 = list(params)
            p0[i] -= DTHETA
            gp0 = gp.copy()
            gp0.params = p0

            p1 = list(params)
            p1[i] += DTHETA
            gp1 = gp.copy()
            gp1.params = p1

            approx_jac[i] = approx_deriv(
                gp0.lh, gp1.lh, DTHETA)

        assert allclose(jac, approx_jac)

    count_failures(check_dlh, N_BIG)


def test_d2lh():
    def check_d2lh(gp):
        params = gp.params.copy()
        hess = gp.d2lh_dtheta2

        approx_hess = np.empty(hess.shape)
        for i in xrange(len(params)):
            p0 = list(params)
            p0[i] -= DTHETA
            gp0 = gp.copy()
            gp0.params = p0

            p1 = list(params)
            p1[i] += DTHETA
            gp1 = gp.copy()
            gp1.params = p1

            approx_hess[:, i] = approx_deriv(
                gp0.dlh_dtheta, gp1.dlh_dtheta, DTHETA)

        assert allclose(hess, approx_hess)

    count_failures(check_d2lh, N_BIG)


def test_dm():
    xo = make_xo()

    def check_dm(gp):
        params = gp.params.copy()
        jac = gp.dm_dtheta(xo)

        approx_jac = np.empty(jac.shape)
        for i in xrange(len(params)):
            p0 = list(params)
            p0[i] -= DTHETA
            gp0 = gp.copy()
            gp0.params = p0

            p1 = list(params)
            p1[i] += DTHETA
            gp1 = gp.copy()
            gp1.params = p1

            approx_jac[i] = approx_deriv(
                gp0.mean(xo), gp1.mean(xo), DTHETA)

        assert allclose(jac, approx_jac)

    count_failures(check_dm, N_BIG)


def test_dtypes():
    gp = make_gp()
    xo = make_xo()

    def check_dtype(x):
        assert x.dtype == DTYPE

    def check_type(x):
        assert type(x) == DTYPE

    yield check_dtype, gp.x
    yield check_dtype, gp.y
    yield check_type, gp.s
    yield check_dtype, gp.params
    yield check_dtype, gp.Kxx
    yield check_dtype, gp.Kxx_J
    yield check_dtype, gp.Kxx_H
    yield check_dtype, gp.Lxx
    yield check_dtype, gp.inv_Lxx
    yield check_dtype, gp.inv_Kxx
    yield check_dtype, gp.inv_Kxx_y
    yield check_type, gp.log_lh
    yield check_type, gp.lh
    yield check_dtype, gp.dloglh_dtheta
    yield check_dtype, gp.dlh_dtheta
    yield check_dtype, gp.d2lh_dtheta2
    yield check_dtype, gp.Kxoxo(xo)
    yield check_dtype, gp.Kxxo(xo)
    yield check_dtype, gp.Kxox(xo)
    yield check_dtype, gp.mean(xo)
    yield check_dtype, gp.cov(xo)
    yield check_dtype, gp.dm_dtheta(xo)


def test_shapes():
    gp = make_gp()
    xo = make_xo()
    n = gp.x.size
    m = xo.size
    n_p = gp.params.size

    def check_prop_ndim(x, ndim):
        assert getattr(gp, x).ndim == ndim

    def check_prop_shape(x, shape):
        assert getattr(gp, x).shape == shape

    def check_func_shape(x, shape):
        assert getattr(gp, x)(xo).shape == shape

    yield check_prop_ndim, 'x', 1
    yield check_prop_shape, 'y', (n,)
    yield check_prop_shape, 'Kxx', (n, n)
    yield check_prop_shape, 'Kxx_J', (n_p-1, n, n)
    yield check_prop_shape, 'Kxx_H', (n_p-1, n_p-1, n, n)
    yield check_prop_shape, 'Lxx', (n, n)
    yield check_prop_shape, 'inv_Lxx', (n, n)
    yield check_prop_shape, 'inv_Kxx', (n, n)
    yield check_prop_shape, 'inv_Kxx_y', (n,)
    yield check_prop_shape, 'dloglh_dtheta', (n_p,)
    yield check_prop_shape, 'dlh_dtheta', (n_p,)
    yield check_prop_shape, 'd2lh_dtheta2', (n_p, n_p)
    yield check_func_shape, 'Kxoxo', (m, m)
    yield check_func_shape, 'Kxxo', (n, m)
    yield check_func_shape, 'Kxox', (m, n)
    yield check_func_shape, 'mean', (m,)
    yield check_func_shape, 'cov', (m, m)
    yield check_func_shape, 'dm_dtheta', (n_p, m)


def test_memoprop_del():
    gp = make_gp()

    def check_del(prop):
        getattr(gp, prop)
        assert prop in gp._memoized
        delattr(gp, prop)
        assert prop not in gp._memoized

    yield check_del, "Kxx"
    yield check_del, "Kxx_J"
    yield check_del, "Kxx_H"
    yield check_del, "Lxx"
    yield check_del, "inv_Lxx"
    yield check_del, "inv_Kxx"
    yield check_del, "inv_Kxx_y"
    yield check_del, "log_lh"
    yield check_del, "lh"
    yield check_del, "dloglh_dtheta"
    yield check_del, "dlh_dtheta"
    yield check_del, "d2lh_dtheta2"


def test_reset_memoized():
    gp = make_gp()

    def check_memoized(prop, val):
        gp.Kxx
        assert gp._memoized != {}
        setattr(gp, prop, val)
        assert gp._memoized == {}

    yield check_memoized, "x", gp.x.copy() + 1
    yield check_memoized, "y", gp.y.copy() + 1
    yield check_memoized, "s", gp.s + 1
    yield check_memoized, "params", gp.params.copy() + 1


def test_set_y():
    gp = make_gp()
    y = gp.y.copy()
    with pytest.raises(ValueError):
        gp.y = y[:, None]

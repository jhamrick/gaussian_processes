import numpy as np
import pytest

from .util import opt, approx_deriv, allclose

EPS = np.finfo(float).eps
DTHETA = opt['dtheta']


######################################################################


def check_params(kernel, params):
    k = kernel(*params)
    assert (k.params == np.array(params)).all()
    k.params = params
    assert (k.params == np.array(params)).all()


def check_invalid_params(kernel, good_params, bad_params):
    with pytest.raises(ValueError):
        kernel(*bad_params)

    k = kernel(*good_params)
    with pytest.raises(ValueError):
        k.params = bad_params


def check_jacobian(k, x):
    kernel = type(k)
    params = k.params.copy()
    jac1 = k.jacobian(x, x)
    jac2 = np.empty_like(jac1)
    k.jacobian(x, x, out=jac2)

    approx_jac = np.empty(jac1.shape)
    for i in xrange(len(params)):
        p0 = list(params)
        p0[i] -= DTHETA
        p1 = list(params)
        p1[i] += DTHETA
        k0 = kernel(*p0)(x, x)
        k1 = kernel(*p1)(x, x)
        approx_jac[i] = approx_deriv(k0, k1, DTHETA)

    assert allclose(jac1, approx_jac)
    assert allclose(jac2, approx_jac)
    assert allclose(jac1, jac2)


def check_dK_dtheta(k, x, p, i):
    kernel = type(k)
    params = k.params.copy()
    f = getattr(k, "dK_d%s" % p)
    dK_dtheta1 = f(x, x)
    dK_dtheta2 = np.empty_like(dK_dtheta1)
    f(x, x, out=dK_dtheta2)

    params0 = list(params)
    params0[i] -= DTHETA
    params1 = list(params)
    params1[i] += DTHETA
    k0 = kernel(*params0)(x, x)
    k1 = kernel(*params1)(x, x)
    approx_dK_dtheta = approx_deriv(k0, k1, DTHETA)

    assert allclose(dK_dtheta1, approx_dK_dtheta)
    assert allclose(dK_dtheta2, approx_dK_dtheta)
    assert allclose(dK_dtheta1, dK_dtheta2)


def check_hessian(k, x):
    kernel = type(k)
    params = k.params.copy()
    hess1 = k.hessian(x, x)
    hess2 = np.empty_like(hess1)
    k.hessian(x, x, out=hess2)

    approx_hess = np.empty(hess1.shape)
    for i in xrange(len(params)):
        p0 = list(params)
        p1 = list(params)
        p0[i] -= DTHETA
        p1[i] += DTHETA
        jac0 = kernel(*p0).jacobian(x, x)
        jac1 = kernel(*p1).jacobian(x, x)
        approx_hess[:, i] = approx_deriv(jac0, jac1, DTHETA)

    assert allclose(hess1, approx_hess)
    assert allclose(hess2, approx_hess)
    assert allclose(hess1, hess2)


def check_d2K_dtheta2(k, x, p1, p2, i):
    kernel = type(k)
    params = k.params.copy()
    f = getattr(k, "d2K_d%sd%s" % (p1, p2))
    d2K_dtheta21 = f(x, x)
    d2K_dtheta22 = np.empty_like(d2K_dtheta21)
    f(x, x, out=d2K_dtheta22)

    params0 = list(params)
    params1 = list(params)
    params0[i] -= DTHETA
    params1[i] += DTHETA
    dK_dtheta0 = getattr(kernel(*params0), "dK_d%s" % p1)(x, x)
    dK_dtheta1 = getattr(kernel(*params1), "dK_d%s" % p1)(x, x)
    approx_d2K_dtheta2 = approx_deriv(dK_dtheta0, dK_dtheta1, DTHETA)

    assert allclose(d2K_dtheta21, approx_d2K_dtheta2)
    assert allclose(d2K_dtheta22, approx_d2K_dtheta2)
    assert allclose(d2K_dtheta21, d2K_dtheta22)

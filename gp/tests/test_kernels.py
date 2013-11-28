import numpy as np
np.seterr(all='raise')

from .util import opt, approx_deriv, allclose

EPS = np.finfo(float).eps
DTHETA = opt['dtheta']


######################################################################


def check_params(kernel, params):
    k = kernel(*params)
    assert (k.params == np.array(params)).all()


def check_jacobian(k, x):
    kernel = type(k)
    params = k.params.copy()
    jac = k.jacobian(x, x)

    approx_jac = np.empty(jac.shape)
    for i in xrange(len(params)):
        p0 = list(params)
        p0[i] -= DTHETA
        p1 = list(params)
        p1[i] += DTHETA
        k0 = kernel(*p0)(x, x)
        k1 = kernel(*p1)(x, x)
        approx_jac[i] = approx_deriv(k0, k1, DTHETA)

    assert allclose(jac, approx_jac)


def check_dK_dtheta(k, x, p, i):
    kernel = type(k)
    params = k.params.copy()
    dK_dtheta = getattr(k, "dK_d%s" % p)(x, x)

    params0 = list(params)
    params0[i] -= DTHETA
    params1 = list(params)
    params1[i] += DTHETA
    k0 = kernel(*params0)(x, x)
    k1 = kernel(*params1)(x, x)
    approx_dK_dtheta = approx_deriv(k0, k1, DTHETA)

    assert allclose(dK_dtheta, approx_dK_dtheta)


def check_hessian(k, x):
    kernel = type(k)
    params = k.params.copy()
    hess = k.hessian(x, x)

    approx_hess = np.empty(hess.shape)
    for i in xrange(len(params)):
        p0 = list(params)
        p1 = list(params)
        p0[i] -= DTHETA
        p1[i] += DTHETA
        jac0 = kernel(*p0).jacobian(x, x)
        jac1 = kernel(*p1).jacobian(x, x)
        approx_hess[:, i] = approx_deriv(jac0, jac1, DTHETA)

    assert allclose(hess, approx_hess)


def check_d2K_dtheta2(k, x, p1, p2, i):
    kernel = type(k)
    params = k.params.copy()
    d2K_dtheta2 = getattr(k, "d2K_d%sd%s" % (p1, p2))(x, x)

    params0 = list(params)
    params1 = list(params)
    params0[i] -= DTHETA
    params1[i] += DTHETA
    dK_dtheta0 = getattr(kernel(*params0), "dK_d%s" % p1)(x, x)
    dK_dtheta1 = getattr(kernel(*params1), "dK_d%s" % p1)(x, x)
    approx_d2K_dtheta2 = approx_deriv(dK_dtheta0, dK_dtheta1, DTHETA)

    assert allclose(d2K_dtheta2, approx_d2K_dtheta2)

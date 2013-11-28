import numpy as np
np.seterr(all='raise')
np.random.seed(2348)

from .util import opt, approx_deriv

EPS = np.finfo(float).eps
THRESH = opt['error_threshold']
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

    diff = jac - approx_jac
    assert (np.abs(diff) < THRESH).all()


# def check_dK_dtheta(kernel, params, x, i):
#     k = kernel(*params)
#     dK_dtheta = k.


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

    diff = hess - approx_hess
    assert (np.abs(diff) < THRESH).all()

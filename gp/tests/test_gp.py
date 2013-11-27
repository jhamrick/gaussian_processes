import numpy as np
from numpy import dot
np.seterr(all='raise')
np.random.seed(2348)

from gp import GP
from gp import GaussianKernel as kernel
from util import opt, rand_params, approx_deriv

DTYPE = np.float64


def make_xy():
    x = np.linspace(-2*np.pi, 2*np.pi, 16).astype(DTYPE)
    y = np.sin(x)
    return x, y


def make_xo():
    xo = np.linspace(-2*np.pi, 2*np.pi, 32).astype(DTYPE)
    return xo


def make_gp():
    x, y = make_xy()
    gp = GP(kernel(1, 1), x, y, s=1)
    return gp


######################################################################

class TestGP(object):

    def __init__(self):
        self.N_big = opt['n_big_test_iters']
        self.N_small = opt['n_small_test_iters']
        self.thresh = 1e-5
        self.dtheta = 1e-5
        self.pfail = 5

    def check_mean(self, gp, y):
        diff = np.abs(gp.mean(gp.x) - y)
        if (diff > self.thresh).any():
            print diff
            raise AssertionError("bad gp mean")

    def check_inv(self, gp):
        I = dot(gp.Kxx, gp.inv_Kxx)
        diff = np.abs(I - np.eye(I.shape[0]))
        if (diff > self.thresh).any():
            print diff
            raise AssertionError("bad inverted kernel matrix")

    def check_dloglh(self, gp, params):
        jac = gp.dloglh_dtheta

        approx_jac = np.empty(jac.shape)
        for i in xrange(len(params)):
            p0 = list(params)
            p0[i] -= self.dtheta
            gp0 = gp.copy()
            gp0.params = p0

            p1 = list(params)
            p1[i] += self.dtheta
            gp1 = gp.copy()
            gp1.params = p1

            approx_jac[i] = approx_deriv(
                gp0.log_lh, gp1.log_lh, self.dtheta)

        diff = np.abs(jac - approx_jac)
        bad = diff > self.thresh
        if bad.any():
            print "threshold:", self.thresh
            print "worst err:", diff.max()
            print "frac bad: ", (np.sum(bad) / float(bad.size))
            print jac
            print approx_jac
            raise AssertionError("bad dloglh_dtheta")

    def check_dlh(self, gp, params):
        jac = gp.dlh_dtheta

        approx_jac = np.empty(jac.shape)
        for i in xrange(len(params)):
            p0 = list(params)
            p0[i] -= self.dtheta
            gp0 = gp.copy()
            gp0.params = p0

            p1 = list(params)
            p1[i] += self.dtheta
            gp1 = gp.copy()
            gp1.params = p1

            approx_jac[i] = approx_deriv(
                gp0.lh, gp1.lh, self.dtheta)

        diff = jac - approx_jac
        bad = diff > self.thresh
        if bad.any():
            print "threshold:", self.thresh
            print "worst err:", diff.max()
            print "frac bad: ", (np.sum(bad) / float(bad.size))
            print jac
            print approx_jac
            raise AssertionError("bad dlh_dtheta")

    def check_d2lh(self, gp, params):
        hess = gp.d2lh_dtheta2

        approx_hess = np.empty(hess.shape)
        for i in xrange(len(params)):
            p0 = list(params)
            p0[i] -= self.dtheta
            gp0 = gp.copy()
            gp0.params = p0

            p1 = list(params)
            p1[i] += self.dtheta
            gp1 = gp.copy()
            gp1.params = p1

            approx_hess[:, i] = approx_deriv(
                gp0.dlh_dtheta, gp1.dlh_dtheta, self.dtheta)

        diff = hess - approx_hess
        bad = diff > self.thresh
        if bad.any():
            print "threshold:", self.thresh
            print "worst err:", diff.max()
            print "frac bad: ", (np.sum(bad) / float(bad.size))
            print hess
            print approx_hess
            raise AssertionError("bad d2lh_dtheta2")

    def check_dm(self, xo, gp, params):
        jac = gp.dm_dtheta(xo)

        approx_jac = np.empty(jac.shape)
        for i in xrange(len(params)):
            p0 = list(params)
            p0[i] -= self.dtheta
            gp0 = gp.copy()
            gp0.params = p0

            p1 = list(params)
            p1[i] += self.dtheta
            gp1 = gp.copy()
            gp1.params = p1

            approx_jac[i] = approx_deriv(
                gp0.mean(xo), gp1.mean(xo), self.dtheta)

        diff = np.abs(jac - approx_jac)
        bad = diff > self.thresh
        if bad.any():
            print "threshold:", self.thresh
            print "worst err:", diff.max()
            print "frac bad: ", (np.sum(bad) / float(bad.size))
            print jac
            print approx_jac
            raise AssertionError("bad dm_dtheta")

    ##################################################################

    def test_mean(self):
        x, y = make_xy()
        failures = 0.
        for i in xrange(self.N_big):
            params = rand_params('h', 'w')
            s = 0
            gp = GP(kernel(*params), x, y, s=s)
            try:
                self.check_mean(gp, y)
            except AssertionError:
                failures += 1
        pfail = 100 * failures / self.N_big
        if pfail > self.pfail:
            print "%.1f%% failed" % pfail
            raise AssertionError

    def test_inv(self):
        x, y = make_xy()
        failures = 0.
        for i in xrange(self.N_small):
            params = rand_params('h', 'w')
            s = rand_params('s')
            gp = GP(kernel(*params), x, y, s=s)
            try:
                self.check_inv(gp)
            except AssertionError:
                failures += 1
        pfail = 100 * failures / self.N_small
        if pfail > self.pfail:
            print "%.1f%% failed" % pfail
            raise AssertionError

    def test_dloglh(self):
        x, y = make_xy()
        failures = 0.
        for i in xrange(self.N_big):
            params = rand_params('h', 'w')
            s = rand_params('s')
            gp = GP(kernel(*params), x, y, s=s)
            try:
                self.check_dloglh(gp, params + (s,))
            except AssertionError:
                failures += 1
        pfail = 100 * failures / self.N_big
        if pfail > self.pfail:
            print "%.1f%% failed" % pfail
            raise AssertionError

    def test_dlh(self):
        x, y = make_xy()
        failures = 0.
        for i in xrange(self.N_big):
            params = rand_params('h', 'w')
            s = rand_params('s')
            gp = GP(kernel(*params), x, y, s=s)
            try:
                self.check_dlh(gp, params + (s,))
            except AssertionError:
                failures += 1
        pfail = 100 * failures / self.N_big
        if pfail > self.pfail:
            print "%.1f%% failed" % pfail
            raise AssertionError

    def test_d2lh(self):
        x, y = make_xy()
        failures = 0.
        for i in xrange(self.N_big):
            params = rand_params('h', 'w')
            s = rand_params('s')
            gp = GP(kernel(*params), x, y, s=s)
            try:
                self.check_d2lh(gp, params + (s,))
            except AssertionError:
                failures += 1
        pfail = 100 * failures / self.N_big
        if pfail > self.pfail:
            print "%.1f%% failed" % pfail
            raise AssertionError

    def test_dm(self):
        x, y = make_xy()
        xo = make_xo()
        failures = 0.
        for i in xrange(self.N_big):
            params = rand_params('h', 'w')
            s = rand_params('s')
            gp = GP(kernel(*params), x, y, s=s)
            try:
                self.check_dm(xo, gp, params + (s,))
            except AssertionError:
                failures += 1
        pfail = 100 * failures / self.N_big
        if pfail > self.pfail:
            print "%.1f%% failed" % pfail
            raise AssertionError

    def test_x_dtype(self):
        gp = make_gp()
        assert gp.x.dtype == DTYPE

    def test_y_dtype(self):
        gp = make_gp()
        assert gp.y.dtype == DTYPE

    def test_s_dtype(self):
        gp = make_gp()
        assert type(gp.s) == DTYPE

    def test_params_dtype(self):
        gp = make_gp()
        assert gp.params.dtype == DTYPE

    def test_Kxx_dtype(self):
        gp = make_gp()
        assert gp.Kxx.dtype == DTYPE

    def test_Kxx_J_dtype(self):
        gp = make_gp()
        assert gp.Kxx_J.dtype == DTYPE

    def test_Kxx_H_dtype(self):
        gp = make_gp()
        assert gp.Kxx_H.dtype == DTYPE

    def test_Lxx_dtype(self):
        gp = make_gp()
        assert gp.Lxx.dtype == DTYPE

    def test_inv_Lxx_dtype(self):
        gp = make_gp()
        assert gp.inv_Lxx.dtype == DTYPE

    def test_inv_Kxx_dtype(self):
        gp = make_gp()
        assert gp.inv_Kxx.dtype == DTYPE

    def test_inv_Kxx_y_dtype(self):
        gp = make_gp()
        assert gp.inv_Kxx_y.dtype == DTYPE

    def test_log_lh_dtype(self):
        gp = make_gp()
        assert type(gp.log_lh) == DTYPE

    def test_lh_dtype(self):
        gp = make_gp()
        assert type(gp.lh) == DTYPE

    def test_dloglh_dtheta(self):
        gp = make_gp()
        assert gp.dloglh_dtheta.dtype == DTYPE

    def test_dlh_dtheta(self):
        gp = make_gp()
        assert gp.dlh_dtheta.dtype == DTYPE

    def test_d2lh_dtheta2(self):
        gp = make_gp()
        assert gp.d2lh_dtheta2.dtype == DTYPE

    def test_Kxoxo_dtype(self):
        xo = make_xo()
        gp = make_gp()
        assert gp.Kxoxo(xo).dtype == DTYPE

    def test_Kxxo_dtype(self):
        xo = make_xo()
        gp = make_gp()
        assert gp.Kxxo(xo).dtype == DTYPE

    def test_Kxox_dtype(self):
        xo = make_xo()
        gp = make_gp()
        assert gp.Kxox(xo).dtype == DTYPE

    def test_mean_dtype(self):
        xo = make_xo()
        gp = make_gp()
        assert gp.mean(xo).dtype == DTYPE

    def test_cov_dtype(self):
        xo = make_xo()
        gp = make_gp()
        assert gp.cov(xo).dtype == DTYPE

    def test_dm_dtheta_dtype(self):
        xo = make_xo()
        gp = make_gp()
        assert gp.dm_dtheta(xo).dtype == DTYPE

    def test_x_shape(self):
        gp = make_gp()
        assert gp.x.ndim == 1

    def test_y_shape(self):
        gp = make_gp()
        n = gp.x.size
        assert gp.y.shape == (n,)

    def test_Kxx_shape(self):
        gp = make_gp()
        n = gp.x.size
        assert gp.Kxx.shape == (n, n)

    def test_Kxx_J_shape(self):
        gp = make_gp()
        n = gp.x.size
        # minus one because we're Kxx_J doesn't include s
        n_p = gp.params.size - 1
        assert gp.Kxx_J.shape == (n_p, n, n)

    def test_Kxx_H_shape(self):
        gp = make_gp()
        n = gp.x.size
        # minus one because we're Kxx_J doesn't include s
        n_p = gp.params.size - 1
        assert gp.Kxx_H.shape == (n_p, n_p, n, n)

    def test_Lxx_shape(self):
        gp = make_gp()
        n = gp.x.size
        assert gp.Lxx.shape == (n, n)

    def test_inv_Lxx_shape(self):
        gp = make_gp()
        n = gp.x.size
        assert gp.inv_Lxx.shape == (n, n)

    def test_inv_Kxx_shape(self):
        gp = make_gp()
        n = gp.x.size
        assert gp.inv_Kxx.shape == (n, n)

    def test_inv_Kxx_y_shape(self):
        gp = make_gp()
        n = gp.x.size
        assert gp.inv_Kxx_y.shape == (n,)

    def test_dloglh_dtheta_shape(self):
        gp = make_gp()
        n_p = gp.params.size
        assert gp.dloglh_dtheta.shape == (n_p,)

    def test_dlh_dtheta_shape(self):
        gp = make_gp()
        n_p = gp.params.size
        assert gp.dlh_dtheta.shape == (n_p,)

    def test_d2lh_dtheta2_shape(self):
        gp = make_gp()
        n_p = gp.params.size
        assert gp.d2lh_dtheta2.shape == (n_p, n_p)

    def test_Kxoxo_shape(self):
        gp = make_gp()
        xo = make_xo()
        m = xo.size
        assert gp.Kxoxo(xo).shape == (m, m)

    def test_Kxxo_shape(self):
        gp = make_gp()
        xo = make_xo()
        n = gp.x.size
        m = xo.size
        assert gp.Kxxo(xo).shape == (n, m)

    def test_Kxox_shape(self):
        gp = make_gp()
        xo = make_xo()
        n = gp.x.size
        m = xo.size
        assert gp.Kxox(xo).shape == (m, n)

    def test_mean_shape(self):
        gp = make_gp()
        xo = make_xo()
        m = xo.size
        assert gp.mean(xo).shape == (m,)

    def test_cov_shape(self):
        gp = make_gp()
        xo = make_xo()
        m = xo.size
        assert gp.cov(xo).shape == (m, m)

    def test_dm_dtheta_shape(self):
        gp = make_gp()
        xo = make_xo()
        m = xo.size
        n_p = gp.params.size
        assert gp.dm_dtheta(xo).shape == (n_p, m)

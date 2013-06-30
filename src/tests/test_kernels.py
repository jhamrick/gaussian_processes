import scipy.stats
import numpy as np
np.seterr(all='raise')

from kernels import GaussianKernel, PeriodicKernel
from util import opt, rand_params, approx_deriv

EPS = np.finfo(float).eps


######################################################################

class TestKernels(object):

    def __init__(self):
        self.N_big = opt['n_big_test_iters']
        self.N_small = opt['n_small_test_iters']
        self.thresh = np.sqrt(EPS) * 10

    def check_params(self, kernel, params):
        k = kernel(*params)
        if k.params != params:
            print k.params
            print params
            raise AssertionError("parameters do not match")

    def check_gaussian_K(self, x, dx, params):
        kernel = GaussianKernel(*params)
        K = kernel(x, x)
        print "Kernel parameters:", kernel.params

        h, w = params
        pdx = scipy.stats.norm.pdf(dx, loc=0, scale=w)
        pdx *= h ** 2

        diff = abs(pdx - K)
        if not (diff < self.thresh).all():
            print self.thresh, diff
            raise AssertionError("invalid gaussian kernel matrix")

    def check_periodic_K(self, x, dx, params):
        kernel = PeriodicKernel(*params)
        K = kernel(x, x)
        print "Kernel parameters:", kernel.params

        h, w, p = params
        pdx = (h ** 2) * np.exp(-2. * (np.sin(dx / (2. * p)) ** 2) / (w ** 2))

        diff = abs(pdx - K)
        if not (diff < self.thresh).all():
            print self.thresh, diff
            raise AssertionError("invalid periodic kernel matrix")

    def check_jacobian(self, kernel, params, x):
        k = kernel(*params)
        jac = k.jacobian(x, x)
        dtheta = self.thresh

        approx_jac = np.empty(jac.shape)
        for i in xrange(len(params)):
            p0 = list(params)
            p0[i] -= dtheta
            p1 = list(params)
            p1[i] += dtheta
            k0 = kernel(*p0)(x, x)
            k1 = kernel(*p1)(x, x)
            approx_jac[i] = approx_deriv(k0, k1, dtheta)

        diff = jac - approx_jac
        bad = np.abs(diff) > self.thresh
        if bad.any():
            print "threshold:", self.thresh
            print "worst err:", np.abs(diff).max()
            print "frac bad: ", (np.sum(bad) / float(bad.size))
            raise AssertionError("bad jacobian")

    def check_hessian(self, kernel, params, x):
        k = kernel(*params)
        hess = k.hessian(x, x)
        dtheta = self.thresh

        approx_hess = np.empty(hess.shape)
        for i in xrange(len(params)):
            p0 = list(params)
            p1 = list(params)
            p0[i] -= dtheta
            p1[i] += dtheta
            jac0 = kernel(*p0).jacobian(x, x)
            jac1 = kernel(*p1).jacobian(x, x)
            approx_hess[:, i] = approx_deriv(jac0, jac1, dtheta)

        diff = hess - approx_hess
        thresh = 1e-4
        bad = np.abs(diff) > thresh
        if bad.any():
            print "threshold:", thresh
            print "worst err:", np.abs(diff).max()
            print "frac bad: ", (np.sum(bad) / float(bad.size))
            raise AssertionError("bad hessian")

    ######################################################################

    def test_gaussian_kernel_params(self):
        for i in xrange(self.N_big):
            params = rand_params('h', 'w')
            yield self.check_params, GaussianKernel, params

    def test_periodic_kernel_params(self):
        for i in xrange(self.N_big):
            params = rand_params('h', 'w', 'p')
            yield self.check_params, PeriodicKernel, params

    def test_gaussian_K(self):
        """Test stats.gaussian_kernel output matrix"""
        x = np.linspace(-2, 2, 10)
        dx = x[:, None] - x[None, :]
        for i in xrange(self.N_big):
            params = rand_params('h', 'w')
            yield (self.check_gaussian_K, x, dx, params)

    def test_periodic_K(self):
        """Test stats.periodic_kernel output matrix"""
        x = np.linspace(-2*np.pi, 2*np.pi, 16)
        dx = x[:, None] - x[None, :]
        for i in xrange(self.N_big):
            params = rand_params('h', 'w', 'p')
            yield (self.check_periodic_K, x, dx, params)

    def test_gaussian_jacobian(self):
        x = np.linspace(-2, 2, 10)
        for i in xrange(self.N_small):
            params = rand_params('h', 'w')
            yield (self.check_jacobian, GaussianKernel, params, x)

    def test_gaussian_hessian(self):
        x = np.linspace(-2, 2, 10)
        for i in xrange(self.N_small):
            params = rand_params('h', 'w')
            yield (self.check_hessian, GaussianKernel, params, x)

    def test_periodic_jacobian(self):
        x = np.linspace(-2*np.pi, 2*np.pi, 16)
        for i in xrange(self.N_small):
            params = rand_params('h', 'w', 'p')
            yield (self.check_jacobian, PeriodicKernel, params, x)

    def test_periodic_hessian(self):
        x = np.linspace(-2*np.pi, 2*np.pi, 16)
        for i in xrange(self.N_small):
            params = rand_params('h', 'w', 'p')
            yield (self.check_hessian, PeriodicKernel, params, x)

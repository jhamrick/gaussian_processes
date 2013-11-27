import numpy as np
np.seterr(all='raise')
np.random.seed(2348)

from gp.kernels import linalg
from gp import GaussianKernel as kernel
from util import opt, rand_params

DTYPE = np.float64


def make_x():
    x = np.linspace(-2*np.pi, 2*np.pi, 16).astype(DTYPE)
    return x


######################################################################

class TestLinAlg(object):

    def __init__(self):
        self.N_big = opt['n_big_test_iters']
        self.N_small = opt['n_small_test_iters']
        self.thresh = 1e-5

    def check_cholesky(self, Kxx):
        Lxx = np.empty_like(Kxx)
        linalg.cholesky(Kxx, Lxx)

        Lxx2 = np.linalg.cholesky(Kxx.astype('f8'))

        diff = np.abs(Lxx - Lxx2)
        if (diff > self.thresh).any():
            print diff
            raise AssertionError("bad cholesky decomposition")

    def test_cholesky(self):
        x = make_x()
        failures = 0.
        for i in xrange(self.N_small):
            params = rand_params('h', 'w')
            K = kernel(*params)
            Kxx = K(x, x).astype('f8').astype(DTYPE)
            try:
                self.check_cholesky(Kxx)
            except AssertionError:
                failures += 1
        pfail = 100 * failures / self.N_small
        if pfail > 0:
            print "%.1f%% failed" % pfail
            raise AssertionError

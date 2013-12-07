import numpy as np


opt = {
    'n_big_test_iters': 100,
    'n_small_test_iters': 10,
    'pct_allowed_failures': 5,
    'error_threshold': 1e-5,
    'dtheta': 1e-5,
    'dtype': np.float64,
    'eps': np.finfo(np.float64).eps,
}


def rand_params(*args):
    params = []
    for param in args:
        if param == 'h':
            params.append(np.random.uniform(0, 2))
        elif param == 'w':
            params.append(np.random.uniform(np.pi / 32., np.pi / 2.))
        elif param == 'p':
            params.append(np.random.uniform(0.33, 3))
        elif param == 's':
            params.append(np.random.uniform(0, 0.5))
        else: # pragma: no cover
            pass
    return tuple(params)


def approx_deriv(y0, y1, dx):
    dy = (y1 - y0) / 2.
    return dy / dx


def make_xy():
    x = np.linspace(-2 * np.pi, 2 * np.pi, 16).astype(opt['dtype'])
    y = np.sin(x)
    return x, y


def make_xo():
    xo = np.linspace(-2 * np.pi, 2 * np.pi, 32).astype(opt['dtype'])
    return xo


def seed():
    np.random.seed(2348)


def allclose(x, y):
    return np.allclose(x, y, rtol=opt['error_threshold'])

import numpy as np


opt = {
    'n_big_test_iters': 20,
    'n_small_test_iters': 5
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
    if len(params) == 1:
        return params[0]
    return tuple(params)


def approx_deriv(y0, y1, dx):
    dy = (y1 - y0) / 2.
    return dy / dx

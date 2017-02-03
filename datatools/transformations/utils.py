import numpy as np


def _gauss(n, m, steepness=5):
    M = np.array([np.arange(0, m)]*n)
    N = np.array([np.arange(0, n)]*m).T

    m0 = m/2
    n0 = n/2

    sm = m/steepness
    sn = n/steepness

    return np.exp(-((M-m0)**2/(2*sm**2)+(N-n0)**2/(2*sn**2)))


def prepare(data, unmask=False, taper=False, avg=False):
    try:
        mask = data.mask
    except AttributeError:
        mask = np.zeros(data.shape) == 1

    if taper:
        weights = _gauss(*data.shape)
    else:
        weights = np.ones(data.shape)

    if avg:
        wm = np.nansum(data*weights)/np.sum(weights[~mask])
        data = (data-wm)*weights
    else:
        data = data*weights

    if isinstance(data, np.ma.MaskedArray) and unmask:
        if avg:
            fill_value = 0
        else:
            fill_value = np.mean(data)
        data = data.filled(fill_value)

    return data

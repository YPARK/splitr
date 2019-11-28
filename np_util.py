# -*- coding: utf-8 -*-

import numpy as np
## from numba import jit

__all__ = ['softmax',
           'softplus',
           'gammaln',
           'RunningAverage']

## @jit(nopython=True)
def logsumexp(a):
    """
    Just the same one in the scipy.special
    """
    a_max = np.max(a, keepdims=True)

    if a_max.ndim > 0:
        a_max[~np.isfinite(a_max)] = 0
    elif not np.isfinite(a_max):
        a_max = 0    
        
    tmp = np.exp(a - a_max)
    s = np.sum(tmp, keepdims=True)
    return np.log(s) + a_max

## @jit(nopython=True)
def softmax(x):
    """
    Just the same one in the scipy.special
    """
    return np.exp(x - logsumexp(x))

## @jit(nopython=True)
def softplus(x) :
    safe = x < 30.
    return np.log1p(np.exp(x * safe)) * safe + x * (1. - safe)

## @jit(nopython=True)
def gammaln(x):
    return - 0.0810614667 - x - np.log(x) + (0.5 + x) * np.log(1.0 + x)

class RunningAverage:
    def __init__(self, _shape):
        """
        """
        self.shape = _shape
        self.clear()

    def clear(self):
        _shape = self.shape
        self.cumsum = np.zeros(_shape)
        self.cumsum_sq = np.zeros(_shape)
        self.n = 0.0

    def __call__(self, new_data):
        self.cumsum += new_data
        self.cumsum_sq += (new_data * new_data)
        self.n += 1.0

    def mean(self):
        assert self.n > 0.0
        return self.cumsum / self.n

    def sd(self):
        assert self.n > 0.0
        mu = self.mean()
        mu2 = self.cumsum_sq / self.n
        return np.sqrt(mu2 - mu*mu)

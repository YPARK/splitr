# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as random

__all__ = ['sample_ess']

def _unif(sz = None):
    return random.uniform(1e-8, 1.0 - 1e-8, size = sz)

def sample_ess(_fun, theta, jitter):
    """

    """
    two_pi_ = 2.0 * np.pi

    fval = _fun(theta)
    fval_target = fval + np.log(_unif())

    phi_ = _unif() * two_pi_
    phi_min = phi_ - two_pi_
    phi_max = phi_

    while True:
        theta_new = np.cos(phi_) * theta + np.sin(phi_) * jitter
        fval = _fun(theta_new)

        if fval >= fval_target: break

        if phi_ > 0.:
            phi_max = phi_
        elif phi_ < 0:
            phi_min = phi_
        else:
            raise Exception("Unstable MCMC: Perhaps not ergodic")

        phi_ = _unif() * (phi_max - phi_min) + phi_min

    theta = theta_new

    return theta, fval


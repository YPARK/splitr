#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

########################
# Import local library #
########################

sys.path.insert(1, os.path.dirname(__file__))

from util import *
from np_util import *
from mcmc import *

def log_lik_nb(
        y         : np.ndarray,   # Data: Bulk expression (m x 1) in R+
        X         : np.ndarray,   # Data: Reference panel (m x cell-types) in R+
        ct_logits : np.ndarray,   # Param: Logits for cell type estimation in R
        log_od    : np.ndarray,   # Param: Over-dispersion for each gene in R
        log_bias  : np.ndarray,   # Param: Gene-level bias in R
        log_lib   : float,        # Param: Library size in R
        a0        : float = 1e-4, # Hyper: Pseudocount in NB
        tau       : float = 1e-2, # Hyper: Overdispersion hyper-parameter
        eps       : float = 1e-8  # Hyper: To prevent numerical zeros 
):
    """
# Data and parameters

    y = m x 1 bulk expression
    X = m x celltype reference expression

    θ = cell-type fraction
    ν = overdispersion parameter
    δ = gene-level bias
    λ = log library size

# Generative model

    π = softmax(θ)
    η = X * π

    β = exp(-λ) / η
    α = softplus(-ν)

    p(y|α,β) = Γ(α + y)/[Γ(y+1)Γ(α)] (1 + 1/β)^(-α) (1 + β)^-y

# Return
    
    log-likelihood value
    """

    pi_  = softmax(ct_logits)
    eta_ = np.dot(X, pi_) + eps
    lam_ = np.exp(log_lib)
    log_beta_ = - log_lib - np.log(eta_) - log_bias

    alpha_ = softplus(- log_od) + a0

    llik_ = (
        gammaln(alpha_ + y + a0)
        - gammaln(alpha_ + a0)
        - gammaln(y + 1.)
        - (alpha_ + a0) * softplus(- log_beta_)
        - y * softplus(log_beta_)
        )

    lprior_ = (
        - (1.0 - tau) * np.log(alpha_)
        - tau / alpha_
        - gammaln(tau)
        + tau * np.log(tau)
    )

    return np.sum(llik_ + lprior_)


################################################################
def deconvolve(
        Y            : np.ndarray,
        X            : np.ndarray,
        nlocal_steps : int = 3,
        nburnin      : int = 100,
        nmcmc        : int = 1000,
        lib_prior_sd : float = 10.0, # library size can be large
        ct_prior_sd  : float = 1.0,  # should maintain unit variance
        od_prior_sd  : float = 10.0, # overdispersion can be high
        bias_prior_sd  : float = 1.0 
):
    """

    Y : bulk data (gene x individual)
    X : reference data (gene x cell type)

    """

    ngene, nind = Y.shape
    _, ncelltype = X.shape

    #########################
    # Initialize parameters #
    #########################

    # local parameters for each individual
    log_lib_ = np.zeros((nind, ))
    logit_ct_ = np.zeros((ncelltype, nind))

    # global parameters shared across individuals
    log_od_ = np.zeros((ngene, ))
    log_bias_ = np.zeros((ngene, ))

    # Running average
    out_logit_ct = RunningAverage(logit_ct_.shape)
    out_log_lib = RunningAverage(log_lib_.shape)
    out_log_od = RunningAverage(log_od_.shape)
    out_log_bias = RunningAverage(log_bias_.shape)

    out_llik = []

    sample_bias_term = nind >= 10

    theta_local_prior = np.array(
        [ct_prior_sd] * ncelltype + [lib_prior_sd]
    )

    theta_od_prior = np.array([od_prior_sd] * ngene)

    theta_bias_prior = np.array([bias_prior_sd] * ngene)

    for epoch in range(nmcmc + nburnin):

        ####################
        # local parameters #
        ####################

        # 1. sample logit cell type fraction and log library size

        llik = 0.

        for i in range(nind):

            theta = np.zeros(ncelltype + 1) # temporary
            theta[:-1] = logit_ct_[:,i]
            theta[-1] = log_lib_[i]
            y = Y[:,i]

            fun_ct_lib = lambda theta_: (
                log_lik_nb(
                    y, X,
                    theta_[:-1],
                    log_od = log_od_,
                    log_bias = log_bias_,
                    log_lib = theta_[-1]
                )
            )

            for j in range(nlocal_steps):
                jitter = random.normal(size=ncelltype + 1) * theta_local_prior
                theta, llik_ = sample_ess(fun_ct_lib, theta, jitter)

            llik += llik_
            logit_ct_[:,i] = theta[:-1]
            log_lib_[i] = theta[-1]

        #####################
        # Global parameters #
        #####################

        # 2. sample log overdispersion

        fun_od = lambda theta_: (
            sum(log_lik_nb(
                Y[:,i], X,
                logit_ct_[:,i],
                log_od = theta_,
                log_bias = log_bias_,
                log_lib = log_lib_[i])
                for i in range(nind))
            )

        theta = log_od_
        jitter = random.normal(size=ngene) * theta_od_prior
        theta, llik = sample_ess(fun_od, theta, jitter)
        log_od_ = theta

        # 3. sample log bias
        if sample_bias_term:
            fun_bias = lambda theta_: (
                sum(log_lik_nb(
                    Y[:,i], X,
                    logit_ct_[:,i],
                    log_od = log_od_,
                    log_bias = theta_,
                    log_lib = log_lib_[i])
                    for i in range(nind))
                )

            theta = log_bias_
            jitter = random.normal(size=ngene) * theta_bias_prior
            theta, llik = sample_ess(fun_bias, theta, jitter)
            log_bias_ = theta

        # Record-keeping
        if epoch >= nburnin:
            out_logit_ct(logit_ct_)
            out_log_lib(log_lib_)
            out_log_od(log_od_)
            if sample_bias_term: out_log_bias(log_bias_)

        out_llik.append(llik)
        _log_msg("MCMC [%04d] [%5.2f]"%(epoch + 1, llik/float(nind*ngene)))

    _log_msg("Done MCMC")

    out = {
        'llik' : np.array(out_llik),
        'logit_ct' : out_logit_ct,
        'log_lib' : out_log_lib,
        'log_od' : out_log_od,
        'log_bias' : out_log_bias,
    }

    return out


# deconv_out = deconvolve(Y, X, nmcmc=1000)



# # construct




# ## from numba import jit

# aa = np.array([1.0] * 7)

# rdir = lambda : np.reshape(random.dirichlet(alpha = aa), (len(aa), 1))

# X = pd.read_csv("ct_ref.txt",sep="\t",header=None)
# X = np.array(X)

# ff = np.concatenate([rdir() for j in range(10)], axis = 1)

# Y = np.dot(X, ff)


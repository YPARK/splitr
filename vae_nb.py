#!/usr/bin/env python3
import sys
import io
import os
import argparse
import datetime
import math

import keras
from keras import backend as K
from keras.layers import Input, Dense, Lambda, Dropout, BatchNormalization, concatenate
from keras.layers import Layer
from keras.regularizers import l2
from keras import losses
from keras import optimizers
from keras.callbacks import Callback

import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix

###################################
# We want everything in this file #
###################################

def _log_msg(msg):
    """
    just print out message with time
    """
    tt = datetime.datetime.now()
    sys.stderr.write("[%s] %s\n"%(tt.strftime("%Y-%m-%d %H:%M:%S"), msg))
    sys.stderr.flush()

###########################
# Keras-related functions #
###########################

def keras_name_func(**kwargs):
    _name = kwargs.get('name',None)
    _name_0 = lambda x: None
    _name_1 = lambda x: '%s_%s'%(_name, str(x))
    return _name_1 if _name is not None else _name_0

def lgamma(x):
    """
    fast approximation of log Gamma function
    """
    return - 0.0810614667 - x - K.log(x) + (0.5 + x) * K.log(1.0 + x)

def softplus(x, cutoff=30.0, min_val=-30.0):
    """
    A safe softplus function for log(exp(x) + 1)

    We want to take the exponential function within (-cutoff, cutoff)
    where cutoff > 0 strictly

    If x > cutoff,            --> x + log(1 + exp(-cutoff)) --> x
    If x < -cutoff,           --> log(1 + exp(-cutoff))     --> 0
    If x in [-cutoff, cutoff] --> log(1 + exp(x))
    """

    overflow = K.cast(K.greater(x, cutoff), K.floatx())
    log1p_exp_pos = K.log(K.exp(K.clip(x, min_val, cutoff)) + 1.0)
    log1p_exp_neg = K.log(K.exp(K.clip(-x, min_val, cutoff)) + 1.0)

    ret = (
        (1.0 - overflow) * log1p_exp_pos +
        (overflow) * (x + log1p_exp_neg)
    )
    return ret

class GaussianStoch(Layer):
    """
    Sampling from Gaussian distribution
    x = [mu, logvar]
    z ~ N(mu, var)
    """
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.min_sig = kwargs.get('min_sig', 1e-8)
        super(GaussianStoch, self).__init__(**kwargs)

    def call(self, x):
        assert isinstance(x, list)
        mu, logvar = x
        batch = K.shape(mu)[0]
        dim = K.int_shape(mu)[1]
        epsilon = K.random_normal(shape=(batch, dim), mean=0.0, stddev=1.0)

        return mu + epsilon * (K.exp(logvar * 0.5) + self.min_sig)

    def build(self, input_shape):
        super(GaussianStoch, self).build(input_shape)


class Log1P(Layer):
    def __init__(self, **kwargs):
        super(Log1P, self).__init__(**kwargs)

    def call(self, x):
        return K.log(x + 1.0)

    def build(self, input_shape):
        super(Log1P, self).build(input_shape)


def gaussian_kl_loss(mu, logvar):
    """
    Gaussian KL-divergence loss function
    output = Eq[log q(z|mu,var)]
    """
    kl_loss = 1 + logvar - K.square(mu) - K.exp(logvar)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    return kl_loss


def add_gaussian_stoch_layer(hh, **kwargs):
    """VAE Gaussian layer

    # arguments
        hh         : the input of this layer

    # options (**kwargs)
        latent_dim : the dimension of hidden units

    # returns
        z_stoch    : latent variables
        mu         : mean variables
        logvar     : log variance variables
    """

    d = K.int_shape(hh)[1]
    latent_dim = kwargs.get('latent_dim', d)

    _name_it = keras_name_func(**kwargs)

    mu = Dense(latent_dim, activation='linear', name=_name_it('mu'))(hh)
    logvar = Dense(latent_dim, activation='linear', name=_name_it('sig'))(hh)
    z_stoch = GaussianStoch(output_dim=(latent_dim,), name=_name_it('stoch'))([mu,logvar])

    return z_stoch, mu, logvar

class ConstBias(Layer):
    """
    Just add constant bias terms for each dimension
    """
    def __init__(self, units, init_val : float = 0.0, **kwargs):
        self.units = units
        self.init_val = init_val
        super(ConstBias, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # This is just to ignore previous
        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            name='zero_kernel',
            initializer='zeros',
            trainable=False
        )

        self.bias = self.add_weight(
            shape=(self.units, ),
            initializer=keras.initializers.constant(self.init_val),
            name='bias',
            trainable=True
        )
        super(ConstBias, self).build(input_shape)

    def call(self, x):
        ret = K.dot(x, self.kernel)
        ret = K.bias_add(ret, self.bias, data_format='channels_last')
        return ret

    def compute_output_shape(self, input_shape):
        ret = list(input_shape)
        ret[-1] = self.units
        return tuple(ret)


###################################
# construct negative binomial VAE #
###################################

class NBLibSize(Layer):
    def __init__(self, **kwargs):
        super(NBLibSize, self).__init__(**kwargs)

    def call(self, x):
        return K.exp(x)

    def build(self, input_shape):
        super(NBLibSize, self).build(input_shape)

class NBOverDisp(Layer):
    def __init__(self, **kwargs):
        self.eps = kwargs.get('eps', 1e-8)
        super(NBOverDisp, self).__init__(**kwargs)

    def call(self, x):
        return 1.0 / (self.eps + softplus(-x))

    def build(self, input_shape):
        super(NBOverDisp, self).build(input_shape)


class NBLogRate(Layer):
    """
    Compute log-rate of the negative binomial model

    # Inputs

    z : batch x d latent state
    λ : batch x 1 library size

    # Parameters

    W : D x d factor-specific gene selection
    δ : 1 x D gene-specific bias

    # Outputs

    μ(i,g) = sum W(g,k) * z(i,k) + λ(i) + δ(g)

    where we give constraints on W s.t. sum_g W(g,k) = 1
    """
    def __init__(self, units, init_val : float = -5.0, **kwargs):
        self.units = units
        self.init_val = init_val
        super(NBLogRate, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        z, _ = input_shape
        d = z[-1]

        self.kernel = self.add_weight(
            shape=(d, self.units),
            name='weight',
            initializer='glorot_normal',
            trainable=True
        )

        self.bias = self.add_weight(
            shape=(self.units, ),
            initializer=keras.initializers.constant(self.init_val),
            name='bias',
            trainable=True
        )

        self.ones_D = K.ones((1, self.units))

        super(NBLogRate, self).build(input_shape)

    def call(self, x):
        """
        μ(i,g) = sum W(g,k) * z(i,k) + λ(i) + δ(g)
        """
        assert isinstance(x, list)
        z, lam = x

        W = self.kernel

        ret = K.dot(z, W) + K.dot(lam, self.ones_D)
        ret = K.bias_add(ret, self.bias, data_format='channels_last')
        return ret

    def weights(self):
        """
        Take units x genes weight matrix
        """
        return self.get_weights()

    def impute(self, z : np.array):
        """
        Predict gene expression profiles without bias
        """
        assert len(z.shape) is 2
        assert z.shape[1] is self.units
        W = self.get_weights()
        return np.dot(z, W)

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_z, _ = input_shape
        return (shape_z[0], self.units)


def nb_loss(x_obs, x_params, D: int, a0: float = 1e-8):
    """
    log-likelihood loss for the negative bignomial model

    # Inputs

    x_obs    : Observed data (#samples x D)
    x_params : Concatenated(μ, ν)
    D        : dimensionality

    # Details

    p(x|α,β) = Γ(α + x + α0)/[Γ(x+1)Γ(α + α0)] (1 + 1/β)^-(α + α0) (1 + β)^-x

    log_lik  = log Γ(α + x + α0) - log Γ(α + α0)
               -(α + α0) * log(1 + 1/β)
               -x * log(1 + β)

             = log Γ(α + x + α0) - log Γ(α + α0)
               -(α + α0) * softplus(μ)
               -x * softplus(-μ)

    mean[x]  = (α + α0) / β
    var[x]   = (α + α0) / β (1 + 1/ β)

    β        = exp(-μ)
    α        = softplus(-ν)

    """

    n_params = K.int_shape(x_params)[1]

    x_mu = x_params[:, :D]
    x_nu = x_params[:, D:(2*D)]
    alpha = softplus(-x_nu)

    # negative log-likelihood

    ret = (
        lgamma(alpha + a0)
        + lgamma(x_obs + 1.0)
        - lgamma(alpha + a0 + x_obs)
        + (alpha + a0) * softplus(x_mu)
        + x_obs * softplus(-x_mu)
    )

    return K.sum(ret, -1)

###########################
# negative log-likelihood #
###########################

def build_nb_model(D, dims_encoding, dims_encoding_lib, **kwargs):
    """
    Build a negative binomial VAE

    # arguments

        D                 : dimensionality of input
        dims_encoding     : a list of dimensions for encoding layers
        dims_encoding_lib : a list of dimensions for encoding layers

    # options

        l2_penalty        : L2 penalty for the mean model (1e-4)
        nu_l2_penalty     : L2 penalty for the variance model (1e-4)
        nn_dropout_rate   : neural network dropout rate (0.1)

    # returns

        model             : a VAE model for training
        latent_mean       : mean of the latent model
        latent_logvar     : standard deviation of the latent model
        library_size      : a model that outputs library size

    """

    l2_penalty = kwargs.get('l2_penalty', 1e-4)
    l2_penalty_nu = kwargs.get('nu_l2_penalty', 1e-4)
    nn_dropout = kwargs.get('nn_dropout_rate', .0)
    nu_bias_init = kwargs.get('dispersion_bias', -4.0)

    x_in = Input(shape=(D,))
    x_log = Log1P()(x_in)

    ###########################
    # 1. Path from x to mean  #
    # Approximation of q(z|x) #
    ###########################

    hh = x_log

    for i,d in enumerate(dims_encoding):

        hh = BatchNormalization(name = 'batch_norm_%d'%(i+1))(hh)

        hh = Dense(
            d,
            activation='relu',
            activity_regularizer=l2(l2_penalty),
            name='encoding_%d'%(i+1)
        )(hh)

        if nn_dropout > 0.0 and nn_dropout < 1.0:
            _name = 'dropout_%d'%(i+1)
            hh = Dropout(rate=nn_dropout, input_shape=(d,), name=_name)(hh)

    d = dims_encoding[-1]

    _temp = add_gaussian_stoch_layer(hh, latent_dim = d, name = 'z_mu_stoch')

    z_mu, z_mu_mean, z_mu_logvar = _temp

    # Approximation of q(E[x]|z) --> this must be linear
    hh = z_mu

    ##################################
    # 2. Path from x to library size #
    # Approximation of q(lib|x)      #
    ##################################

    hh_lib = x_log

    for i,d in enumerate(dims_encoding_lib):

        hh = BatchNormalization(name = 'lib_batch_norm_%d'%(i+1))(hh)

        hh_lib = Dense(
            d,
            activation='relu',
            activity_regularizer=l2(l2_penalty),
            name='lib_%d'%(i+1)
        )(hh_lib)

        if nn_dropout > 0.0 and nn_dropout < 1.0:
            _name = 'lib_dropout_%d'%(i+1)
            hh_lib = Dropout(rate=nn_dropout, input_shape=(d,), name=_name)(hh_lib)

    d = dims_encoding_lib[-1]

    _temp = add_gaussian_stoch_layer(hh_lib, latent_dim = d, name='z_lib_stoch')
    z_lib, z_lib_mean, z_lib_logvar = _temp

    x_lib = Dense(1, name='x_lib')(z_lib)

    ####################
    # combine log rate #
    ####################

    log_rate_layer = NBLogRate(D, name="NBLogRate")
    x_mu = log_rate_layer([z_mu, x_lib])

    #############################
    # 3. Path form x to q(nu|x) #
    #############################

    x_nu = ConstBias(
        D,
        init_val = nu_bias_init,
        name='x_nu'
    )(hh_lib)

    #################################
    # synthesize the training model #
    #################################

    x_out = concatenate([x_mu, x_nu], axis=-1, name='x_out')

    model = keras.Model(x_in, x_out)

    #########################
    # latent variable model #
    #########################

    latent_mu = keras.Model(x_in, z_mu_mean)

    latent_logvar = keras.Model(x_in, z_mu_logvar)

    ################
    # library size #
    ################

    x_lib_out = NBLibSize(
        name='out_lib'
    )(x_lib)

    libsize = keras.Model(x_in, x_lib_out)

    def vae_loss(_weight):

        # define composite loss function
        def loss(x, xhat):
            lik_loss = nb_loss(x, xhat, D=D)
            kl_mu = gaussian_kl_loss(z_mu_mean, z_mu_logvar)
            kl_lib = gaussian_kl_loss(z_lib_mean, z_lib_logvar)
            return _weight * (kl_mu + kl_lib) + lik_loss

        return loss # return this composite loss

    return model, vae_loss, latent_mu, latent_logvar, libsize

################
# optimization #
################

class KLAnnealing(Callback):

    def __init__(self, weight, base : float = 1e-4, speed : float = 100.):
        self.base = base
        self.speed = speed
        self.weight = weight

    def on_epoch_end(self, epoch, logs={}):
        t = float(epoch) / self.speed
        base = self.base
        new_weight = base + (1.0 - base) * (1.0 - math.exp(-t))

        K.set_value(self.weight, new_weight)
        _log_msg('Adjusted KL weight %f'%K.get_value(self.weight))

def train_nb_vae_kl_annealing(_model, _loss, xx, **kwargs):

    kl_weight = K.variable(0.0)

    lr = kwargs.get('learning_rate', 1e-4)

    opt = optimizers.adam(lr=lr, clipvalue=.01)

    _model.compile(
        optimizer=opt,
        loss = _loss(kl_weight)
    )

    trace = _model.fit(
        xx,
        xx,
        callbacks=[KLAnnealing(kl_weight)],
        **kwargs
    )

    return trace


#######################
# commandline routine #
#######################

if __name__ == '__main__':

    import argparse

    _desc = r"""
    Train Negative Binomial VAE
"""

    parser = argparse.ArgumentParser(
        description=_desc,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--data', default=None)
    parser.add_argument('--dims_latent', default="128,1024,16")
    parser.add_argument('--dims_library', default="32,4")
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    _dims_latent = list(map(int, args.dims_latent.split(',')))
    _dims_library = list(map(int, args.dims_library.split(',')))

    if args.data is None:
        _log_msg("Need data to fit the model!")
        exit(1)

    _log_msg("Need to implement")

    exit(1)


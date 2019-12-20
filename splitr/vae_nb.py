#!/usr/bin/env python3
import sys
import os
import argparse
import datetime
import math
import numpy as np

########################
# Import local library #
########################

sys.path.insert(1, os.path.dirname(__file__))

from util import *
from scio import *
from keras_vae import *
from np_util import gammaln as _lgamma

from keras.utils import plot_model

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
        self.a0 = kwargs.get('a0', 1e-8)
        super(NBOverDisp, self).__init__(**kwargs)

    def call(self, x):
        return 1.0 / (self.a0 + softplus(-x))

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
    def __init__(
            self,
            units,
            init_val : float = -5.0,
            max_log_lib : float = 15.0,
            **kwargs):
        self.units = units
        self.init_val = init_val
        self.max_log_lib = max_log_lib
        super(NBLogRate, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        z, _ = input_shape
        d = z[-1]

        self.kernel = self.add_weight(
            shape=(d, self.units),
            name='weight',
            initializer='glorot_uniform',
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

        # To gain stability
        lam = K.clip(lam, 0.0, self.max_log_lib)

        W = self.kernel
        ret = K.dot(z, W) + K.dot(lam, self.ones_D)
        ret = K.bias_add(ret, self.bias, data_format='channels_last')
        return ret

    def weights(self):
        """
        Take units x genes weight matrix
        """
        return K.get_value(self.kernel)

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


def nb_loss(
        x_obs,             # observed data
        x_params,          # estimated parameters
        D: int,            # dimensionality
        a0: float = 1e-4,   # minimum inverse over-dispersion
        tau : float = 1e-2 # hyperparameter for over-dispersion
):
    """
    log-likelihood loss for the negative bignomial model

    # Inputs

    x_obs    : Observed data (#samples x D)
    x_params : Concatenated(μ, ν)
    D        : dimensionality

    a0       : minimum inverse over-dispersion
    tau      : hyperparameter

    # Details

    p(x|α,β) = Γ(α + x)/[Γ(x+1)Γ(α)] (1 + 1/β)^(-α) (1 + β)^-x

    log_lik  = log Γ(α + x) - log Γ(α)
               -α * log(1 + 1/β)
               -x * log(1 + β)

             = log Γ(α + x) - log Γ(α)
               -α * softplus(μ)
               -x * softplus(-μ)

    mean[x]  = α / β
    var[x]   = α / β (1 + 1/ β)

    β        = exp(-μ)
    α        = softplus(-ν)

    # Additional prior

    1/α      ~ Gamma(τ, τ)

    Ε(1/α)   = τ / τ = 1
    V(1/α)   = 1 / τ

    ln_prior = (1-τ) ln(α) - τ/α
               - log Γ(τ) + τ log(τ)
    """

    n_params = K.int_shape(x_params)[1]

    x_mu = x_params[:, :D]
    x_nu = x_params[:, D:(2*D)]

    alpha = softplus(-x_nu) + a0
    alpha = K.clip(alpha, a0, 1e4)

    neg_log_prior = (
        (tau - 1.0) * K.log(alpha)
        + tau / alpha
        + _lgamma(tau)
        - tau * math.log(tau)
    )

    neg_log_lik = (
        lgamma(alpha)
        + lgamma(x_obs + 1.0)
        - lgamma(alpha + x_obs)
        + alpha * softplus(x_mu)
        + x_obs * softplus(-x_mu)
    )

    ret = neg_log_lik + neg_log_prior
    return K.sum(ret, -1)

###########################
# negative log-likelihood #
###########################

def build_nb_model(
        D,
        Dc,
        dims_encoding,
        dims_encoding_lib,
        iaf_trans,
        **kwargs
):
    """
    Build a negative binomial VAE

    # arguments

        D                 : dimensionality of input
        Dc                : dimensionality of covariates
        dims_encoding     : a list of dimensions for encoding layers
        dims_encoding_lib : a list of dimensions for library encoding layers

    # options

        l1_penalty        : L1 penalty for the mean model (1e-2)
        nu_l1_penalty     : L1 penalty for the variance model (1e-2)
        nn_dropout_rate   : neural network dropout rate (0.1)

        a0                : minimum inverse over-dispersion
        tau               : hyperparameter

    # returns

        model             : a training model
        vae_loss          : a loss function
        latent_models     : a dictionary of latent models

    """

    l1_penalty = kwargs.get('l1_penalty', 1e-2)
    l1_penalty_nu = kwargs.get('nu_l1_penalty', 1e-2)
    nn_dropout = kwargs.get('nn_dropout_rate', .0)
    nu_bias_init = kwargs.get('dispersion_bias', -0.0)
    _max_log_lib = math.log(kwargs.get('lib_target', 1e4))
    iaf_concat = kwargs.get('iaf_concat', True)

    x_in = Input(shape=(D,))
    x_log = Log1P()(x_in)

    #############################
    # 1. A path from x to mean  #
    # Approximation of q(z|x)   #
    #############################

    hh = x_log

    for i,d in enumerate(dims_encoding):

        hh = Dense(
            d,
            activation='relu',
            activity_regularizer=l1(l1_penalty),
            name='mu_%d'%(i+1)
        )(hh)

        hh = BatchNormalization(name = 'mu_batch_norm_%d'%(i+1))(hh)

        if nn_dropout > 0.0 and nn_dropout < 1.0:
            _name = 'mu_dropout_%d'%(i+1)
            hh = Dropout(rate=nn_dropout, input_shape=(d,), name=_name)(hh)

    d = dims_encoding[-1]

    if iaf_trans > 0:

        IAF = build_iaf_stack(
            hh,
            _name="IAF_MU",
            latent_dim = d,
            num_trans = iaf_trans,
            concat_h_z = iaf_concat
        )
        z_mu, z_mu_mean, z_mu_logvar, kl_mu  = IAF

    else:
        _temp = add_gaussian_stoch_layer(
            hh,
            latent_dim = d,
            name = 'z_mu_stoch'
        )
        z_mu, z_mu_mean, z_mu_logvar = _temp
        kl_mu = gaussian_kl_loss(z_mu_mean, z_mu_logvar)

    ##################
    # spike-and-slap #
    ##################

    gumbel_temperature = K.variable(1.0, name="Gumbel Temperature")
    latent_spike = None

    if kwargs.get('with_spike', False):

        z_spike_logits = hh

        z_spike = BinaryGumbelSoftmax(
            output_dim = (d,),
            temperature = gumbel_temperature,
            name = "spike_z"
        )(z_spike_logits)

        z_mu = layers.multiply([z_mu , z_spike])

        latent_spike = keras.Model(x_in, z_spike)

        _log_msg("With spike-slab latent states")

    ##############################################
    # An additional path from covariates to mean #
    ##############################################

    x_covar = Input(shape=(Dc,))

    z_covar  = Dense(
        units = dims_encoding[-1],
        name = 'z_covar',
        activity_regularizer = l1(l1_penalty)
    )(x_covar)

    z_mu_covar = layers.add([z_mu, z_covar], name='z_mu_covar')

    ##################################
    # 2. Path from x to library size #
    # Approximation of q(lib|x)      #
    ##################################

    hh_lib = x_log

    for i,d in enumerate(dims_encoding_lib):

        hh_lib = Dense(
            d,
            activation='relu',
            activity_regularizer=l1(l1_penalty),
            name='lib_%d'%(i+1)
        )(hh_lib)

        hh_lib = BatchNormalization(name = 'lib_batch_norm_%d'%(i+1))(hh_lib)

        if nn_dropout > 0.0 and nn_dropout < 1.0:
            _name = 'lib_dropout_%d'%(i+1)
            hh_lib = Dropout(rate=nn_dropout, input_shape=(d,), name=_name)(hh_lib)

    d = dims_encoding_lib[-1]

    _temp = add_gaussian_stoch_layer(hh_lib, latent_dim = d, name='z_lib_stoch')
    z_lib, z_lib_mean, z_lib_logvar = _temp
    kl_lib = gaussian_kl_loss(z_lib_mean, z_lib_logvar)

    x_lib_l1 = 1.0

    _log_msg("apply strong L1 on library size estimation: %f"%x_lib_l1)

    x_lib = Dense(
        units = 1,
        name = 'x_lib',
        activity_regularizer=l1(x_lib_l1)
    )(z_lib)

    ####################
    # combine log rate #
    ####################

    log_rate_layer = NBLogRate(
        D,
        name="NBLogRate",
        max_log_lib=_max_log_lib
    )
    x_mu = log_rate_layer([z_mu_covar, x_lib])

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

    x_out = layers.concatenate(
        [x_mu, x_nu],
        axis=-1,
        name='x_out')

    model = keras.Model([x_in, x_covar], x_out)

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

    model.kl_weight = K.variable(0.0, name = "KL weight")

    model.gumbel_temperature = gumbel_temperature

    kl_spike = 0.0

    if kwargs.get('with_spike', False):
        kl_spike = binary_gumbel_kl_loss(z_spike, z_spike_logits)

    a0 = kwargs.get('a0', 1e-2)  # minimum inverse over-dispersion
    tau = kwargs.get('tau', 1e-2)# hyperparameter

    def vae_loss(_weight):
        # define composite loss function
        def loss(x, xhat):
            lik_loss = nb_loss(x, xhat, D = D, a0 = a0, tau = tau)
            return _weight * (kl_mu + kl_lib + kl_spike) + lik_loss

        return loss # return this composite loss

    latent_models = {
        'mu'        : latent_mu,
        'mu_logvar' : latent_logvar,
        'spike'     : latent_spike,
        'library'   : libsize
    }

    return model, vae_loss, latent_models

################
# optimization #
################

class Annealing(Callback):

    def __init__(
            self,
            kl_weight,
            gumbel_temperature,
            gumbel_rate : float = 1e-2,
            kl_base : float = 1e-2,
            kl_rate : float = 1.0
    ):
        self.kl_base = kl_base
        self.kl_rate = kl_rate
        self.kl_weight = kl_weight

        self.gumbel_temperature = gumbel_temperature
        self.gumbel_rate = gumbel_rate

    def on_epoch_end(self, epoch, logs={}):

        ###########################
        # KL divergence annealing #
        ###########################

        t = float(epoch) * self.kl_rate
        kl_base = self.kl_base
        new_kl_weight = kl_base + (1.0 - kl_base) * (1.0 - math.exp(-t))

        K.set_value(self.kl_weight, new_kl_weight)

        ##############################
        # Gumbel Softmax temperature #
        ##############################

        t = float(epoch) * self.gumbel_rate
        K.set_value(self.gumbel_temperature, max(0.1, math.exp(-t)))

        _log_msg('Adjust KL divergence by  %.2e'%K.get_value(self.kl_weight))
        _log_msg('Apply Gumbel temperature %.2e'%K.get_value(self.gumbel_temperature))


def train_nb_vae_kl_annealing(_model, _loss, xx_cc : list, **kwargs):
    """
    Train VAE with KL divergence annealing
    """

    xx, cc = xx_cc

    kl_weight = _model.kl_weight
    gumbelT = _model.gumbel_temperature

    K.set_value(kl_weight, 0.0)
    K.set_value(gumbelT, 1.0)

    lr = kwargs.get('learning_rate', 1e-2)
    clipval = kwargs.get('clipvalue', lr)
    epochs = kwargs.get('epochs', 100)

    _gumbel_rate = kwargs.get('gumbel_rate', 1e-2)
    _kl_rate = kwargs.get('kl_rate', 1.0)

    opt = optimizers.adam(lr=lr, clipvalue=clipval)

    _model.compile(
        optimizer=opt,
        loss = _loss(kl_weight)
    )

    trace = _model.fit(
        xx_cc,
        xx,
        epochs = epochs,
        callbacks=[Annealing(kl_weight, gumbelT, kl_rate=_kl_rate, gumbel_rate = _gumbel_rate)]
    )

    return trace

#######################
# commandline routine #
#######################

def standardize_columns(C) :

    covar = np.array(C)
    covar = np.ma.masked_where(~np.isfinite(covar), covar)
    covar_std = (covar - np.mean(covar, 0)) / np.std(covar, 0)
    ret = covar_std.data
    ret[~np.isfinite(ret)] = 0.0

    return ret

def preprocess_data(args):

    X = read_mtx_file(args.data)

    _log_msg("Read data matrix : %s"%args.data)

    if args.columns_are_samples:
        X = X.T

    if args.covar is not None:
        C = read_mtx_file(args.covar)
        _log_msg("Read covariate matrix : %s"%args.covar)

        if args.columns_are_samples:
            C = C.T

        if args.standardize_covar:
            C = standardize_columns(C)
            _log_msg("Standardization of the covariate matrix")

        C[~np.isfinite(C)] = 0.0

    else:
        C = np.array(np.zeros((X.shape[0], 1)))
        _log_msg("No covariate")

    if X.shape[0] != C.shape[0]:
        raise Exception("X and C contain different # of samples")

    ############
    # data Q/C #
    ############

    xx, rows, cols = filter_zero_rows_cols(
        X,
        args.sample_cutoff,
        args.feature_cutoff
    )

    cc = C[rows, :]

    if args.standardize_data:
        xx = normalize_X(xx, args.std_target)
    else:
        xx[~np.isfinite(xx)] = 0.0

    return xx, cc, rows, cols

def run(args):

    _dims_latent = list(map(int, args.dlatent.split(',')))
    _dims_library = list(map(int, args.dlibrary.split(',')))

    xx, cc, rows, cols = preprocess_data(args)

    #######################
    # Construct the model #
    #######################

    _log_msg("Building a model")
    _log_msg("Min 1/overdispersion: %.2e"%args.a0)

    model, vae_loss, latent_models = build_nb_model(
        D = xx.shape[1],
        Dc = cc.shape[1],
        dims_encoding = _dims_latent,
        dims_encoding_lib = _dims_library,
        with_spike = args.spike,
        nn_dropout_rate = args.dropout,
        a0 = args.a0,
        lib_target = args.std_target,
        iaf_trans = args.iaf_trans
    )

    plot_model(model, args.out + "_model.png")

    _log_msg("Start training the NB-VAE model")

    out = train_nb_vae_kl_annealing(
        model,
        vae_loss,
        [xx, cc],
        learning_rate = args.learning_rate,
        clipvalue = args.clip_value,
        epochs = args.epochs,
        batch_size = args.batch,
        kl_rate=args.kl_rate
    )

    _log_msg("Successfully finished")

    model.X = xx
    model.C = cc
    model.rows = rows
    model.cols = cols

    trace = out.history.get('loss', [])

    _log_msg("Attached the training data to the model")

    return model, latent_models, trace


def write_results(_model, _latent_models, trace, args) :

    _log_msg("Output latent representations")

    X = _model.X
    C = _model.C
    rows = _model.rows
    cols = _model.cols

    z_mean = _latent_models['mu'].predict(X)
    z_logvar = _latent_models['mu_logvar'].predict(X)
    z_library = _latent_models['library'].predict(X)
    z_spike = np.array([[]])
    if _latent_models['spike'] is not None:
        z_spike = _latent_models['spike'].predict(X)

    weight = _model.get_layer("NBLogRate").weights()

    save_array(z_mean, args.out + ".z_mean.gz")
    save_array(z_logvar, args.out + ".z_logvar.gz")
    save_array(z_library, args.out + ".z_library.gz")
    save_array(z_spike, args.out + ".z_spike.gz")

    _log_msg("Output other results")

    save_list(trace, args.out + ".elbo")
    save_array(weight, args.out + ".weights.gz")
    save_list(rows.tolist(), args.out + ".samples.gz")
    save_list(cols.tolist(), args.out + ".features.gz")

    return


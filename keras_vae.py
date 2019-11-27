###########################
# Keras-related functions #
###########################

import keras
from keras import backend as K
from keras.layers import Input, Dense, Lambda, Dropout, BatchNormalization
from keras.layers import Layer
from keras import layers
from keras.regularizers import l2
from keras import losses
from keras import optimizers
from keras.callbacks import Callback

################################################################################
# math functions not fully implemented in the current version of Keras backend #
################################################################################

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

def logsumexp(logits, axis=-1):
    """
    log(sum(exp(logits)))
    """
    d = K.int_shape(logits)[-1]
    _max = K.max(logits, axis=axis)
    _max_prop = K.dot(K.max(logits, axis=axis, keepdims=True), K.ones((1,d)))
    return K.log(K.sum(K.exp(logits - _max_prop), axis=axis)) + _max

#############################
# Gaussian stochastic layer #
#############################

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

###############################
# inverse auto-regressive VAE #
###############################

class TransformIAF(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(TransformIAF, self).__init__(**kwargs)

    def call(self, x):
        assert isinstance(x, list)
        mu, sig, z_prev = x
        return mu * (1.0 - sig) + z_prev * sig

    def build(self, input_shape):
        super(TransformIAF, self).build(input_shape)

def add_iaf_transformation(hh, z_prev, **kwargs):
    """Inverse Autoregressive Flow (a helper function)
    """

    rank = kwargs.get('rank',None)
    act = kwargs.get('act','linear')
    pmin = kwargs.get('pmin',1e-2)
    l2_penalty = kwargs.get('l2_penalty', 1e-4)

    _name_it = keras_name_func(**kwargs)

    d = K.int_shape(z_prev)[1]

    ## We want new proposal (mu) to be independent of the previous ones
    ## hh_z = concatenate([hh, z_prev], axis=1, name=_name_it('iaf_concat'))

    mu = Dense(
        d,
        activation=act,
        activity_regularizer=l2(l2_penalty),
        name=_name_it('mu')
    )(hh)
    
    sig = Dense(
        d,
        activation=KMath.sigmoid,
        activity_regularizer=l2(l2_penalty),
        name=_name_it('sig')
    )(hh)

    z_next = TransformIAF(output_dim=(d,), name=_name_it('z'))([mu,sig,z_prev])

    kl_loss = K.sum(K.log(sig), axis=-1)
    return z_next, mu, sig, kl_loss

def build_iaf_stack(input_hidden, _name, **kwargs):
    """Inverse Autoregressive Flow

    # arguments
        input_hidden : the input of IAF layer
        _name        : must give a name
        num_trans    : number of transformations

    # options (**kwargs)
        latent_dim   : the dimension of hidden units
        pmin         : the minimum of gating functions (default: 1e-4)

    # returns
        z_layers     : latent variables
        mu_layers    : mean change variables
        sig_layers   : variance gating variables
        kl_loss      : KL divergence loss
    """

    _stoch = add_gaussian_stoch_layer(input_hidden, name=_name, **kwargs)
    z, mu0, sig0, kl_loss = _stoch

    z_layers = [z]
    mu_layers = [mu0]
    sig_layers = [sig0]

    num_trans = kwargs.get('num_trans', 1)

    for l in range(1, num_trans + 1):
        hdr = '%s_%d'%(_name, l)
        z, mu, sig, _kl = add_iaf_transformation(input_hidden, z, name=hdr, **kwargs)
        z_layers.append(z)
        mu_layers.append(mu)
        sig_layers.append(sig)
        kl_loss -= _kl

    return z_layers, mu_layers, sig_layers, kl_loss

##################
# Gumbel softmax #
##################

class GumbelSoftmax(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.temperature = kwargs.get('temperature', K.variable(0.1))
        self.eps = kwargs.get('eps', 1e-4)
        self.num_clusters = kwargs.get('num_clusters', output_dim)
        super(GumbelSoftmax, self).__init__(**kwargs)

    def call(self, logits):
        batch = K.shape(logits)[0]
        dim = K.int_shape(logits)[1]
        eps = self.eps
        tt = K.get_value(self.temperature)
        u = K.random_uniform(shape=(batch, dim), minval=eps, maxval=1.0-eps)
        g = -K.log(-K.log(u))
        return K.softmax((logits + g)/tt)

    def build(self, input_shape):
        super(GumbelSoftmax, self).build(input_shape)

def add_gumbel_softmax_layer(hh, **kwargs):
    """VAE Gumbel-Softmax layer
    """

    _name_it = keras_name_func(**kwargs)

    num_clusters = kwargs.get('num_clusters', K.int_shape(hh)[1])

    logits = Dense(num_clusters, name=_name_it('logits'))(hh)

    z_stoch = GumbelSoftmax(
        output_dim=(num_clusters, ),
        name=_name_it('softmax')
    )(logits)

    return z_stoch, logits

def gumbel_kl_loss(z_stoch, logits):
    num_clusters = K.int_shape(logits)[1]
    _lnK = np.log(float(num_clusters))
    kl_loss = K.sum(z_stoch * _lnK, axis=-1)
    kl_loss += K.sum(z_stoch * logits, axis=-1)
    kl_loss -= logsumexp(logits, axis=-1)
    return kl_loss

#########################
# binary Gumbel softmax #
#########################

class BinaryGumbelSoftmax(Layer):

    def __init__(self, output_dim, temperature, **kwargs):
        self.output_dim = output_dim
        self.temperature = temperature
        self.eps = kwargs.get('eps', 1e-4)

        super(BinaryGumbelSoftmax, self).__init__(**kwargs)

    def call(self, logits):
        """
        z ~ sigmoid((logits + logit(u))/temperature)
        """
        batch = K.shape(logits)[0]
        dim = K.int_shape(logits)[1]
        eps = self.eps
        tt = K.get_value(self.temperature)
        u = K.random_uniform(shape=(batch, dim), minval=eps, maxval=1.0-eps)
        g = K.log(u) - K.log(1.0 - u)
        return K.sigmoid((logits + g)/tt)

    def build(self, input_shape):
        super(BinaryGumbelSoftmax, self).build(input_shape)

def binary_gumbel_kl_loss(z_stoch, logits):
    return K.sum(- z_stoch * (softplus(-logits) + np.log(2.0)), axis=-1)

#########
# misc. #
#########

def create_ones(xx, dd=1):
    """
    Create 1's bypassing the input xx

    xx : input variable
    dd : dimension of 1's
    """

    dim = K.int_shape(xx)[1]
    ret = K.abs(K.dot(xx, K.ones((dim, dd))))
    ret = (ret + 1.0)/(ret + 1.0)
    return ret


class LogRelu(Layer):
    def __init__(self, units, init_val : float = 0.0, **kwargs):
        self.units = units
        self.init_val = init_val
        super(LogRelu, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
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
        super(LogRelu, self).build(input_shape)

    def call(self, x):
        ret = K.dot(K.log(x + 1.0), self.kernel)
        ret = K.bias_add(ret, self.bias, data_format='channels_last')
        ret = K.relu(ret)
        return ret

    def compute_output_shape(self, input_shape):
        ret = list(input_shape)
        ret[-1] = self.units
        return tuple(ret)

class WeightedSum(Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        super(WeightedSum, self).__init__(**kwargs)

    def call(self, x):
        assert isinstance(x, list)
        aa, bb = x
        k = K.int_shape(bb)[1]
        d = self.units
        m = K.shape(aa)[0]

        # aa = m x (d * k) and bb = m x k
        _aa = K.reshape(aa, (m, d, k))
        return K.batch_dot(_aa, bb, axes=(2, 1))

    def build(self, input_shape):
        super(WeightedSum, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return (shape_a[0], self.units)


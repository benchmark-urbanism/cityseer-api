"""
On hyperparameter tuning for beta vae with capacity increase
https://github.com/1Konny/Beta-VAE/issues/8#issuecomment-445126239

Some examples of code implementations:
https://keras.io/examples/variational_autoencoder/
https://tiao.io/post/tutorial-on-variational-autoencoders-with-a-concise-keras-implementation/

See below for detailed comments on vae loss
"""

import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, layers, metrics, models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mse_sum(y_true, y_pred):
    # sum over dimensions
    rl = K.sum(K.square(y_pred - y_true), axis=-1)
    # take mean over samples
    return K.mean(rl)


class DeepLayer(layers.Layer):
    """
    default bias initialiser is zeros
    default weights initialiser is glorot uniform (Xavier)
    He is supposedly better for He but Xavier seems to untangle better
    glorot normal also doesn't work as well as glorot uniform
    """

    def __init__(
        self,
        hidden_layer_dims: tuple = (256, 256, 256),
        activation_func: str = "swish",
        kernel_init: str = "glorot_uniform",
        activity_reg: str = None,
        dropout: float = None,
        seed: int = 0,
        batch_norm: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_layer_dims = hidden_layer_dims
        self.deep_layers = []
        for idx, dim in enumerate(self.hidden_layer_dims):
            self.deep_layers.append(
                layers.Dense(
                    dim,
                    activation=activation_func,
                    kernel_initializer=kernel_init,
                    activity_regularizer=activity_reg,
                )
            )
            if batch_norm:
                self.deep_layers.append(layers.BatchNormalization())
            # add dropout for first layer only - akin to simulating variegated data
            # leave remaining layers to form abstractions
            if dropout is not None:
                self.deep_layers.append(layers.Dropout(rate=dropout, seed=seed))

    def call(self, inputs, training=True, **kwargs):
        X = inputs
        for deep_layer in self.deep_layers:
            X = deep_layer(X, training=training)
        return X

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_layer_dims": self.hidden_layer_dims,
                "dLayers": self.deep_layers,
                "bnLayers": self.bnLayers,
            }
        )
        return config


class SamplingLayer(layers.Layer):
    def __init__(self, latent_dim: int, seed: int, mu_kernel_init: str, **kwargs):
        """
        Z_log_var is used instead of standard deviation to speed up convergence
        Standard deviation can be recovered per exp(0.5 * Z_log_var)
        see page 435 in Hands On Machine Learning
        """
        super().__init__(**kwargs)
        self.seed = seed
        self.latent_dim = latent_dim
        self.Z_mu_layer = layers.Dense(
            self.latent_dim,
            name="Z_mu",
            kernel_initializer=mu_kernel_init,
            activation="linear",
        )
        self.Z_logvar_layer = layers.Dense(
            self.latent_dim,
            name="Z_log_var",
            kernel_initializer=initializers.Zeros(),
            activation="linear",
        )

    def call(self, inputs, training=True, **kwargs):
        batch = K.shape(inputs)[0]
        Z_mu = self.Z_mu_layer(inputs, training=training)
        Z_log_var = self.Z_logvar_layer(inputs, training=training)
        # epsilon variable removes stochastic process from chain of differentiation
        epsilon = K.random_normal(shape=(batch, self.latent_dim), mean=0.0, stddev=1.0, seed=self.seed)
        # see page 5 in auto-encoding variational bayes
        Z = Z_mu + K.exp(0.5 * Z_log_var) * epsilon
        return [Z_mu, Z_log_var, Z]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "seed": self.seed,
                "latent_dim": self.latent_dim,
                "Z_mu_layer": self.Z_mu_layer,
                "Z_logvar_layer": self.Z_logvar_layer,
            }
        )
        return config


class KLDivergenceLayer(layers.Layer):
    """
    custom losses
    https://stackoverflow.com/questions/52034983/
    how-is-total-loss-calculated-over-multiple-classes-in-keras
    https://stackoverflow.com/questions/52172859/
    loss-calculation-over-different-batch-sizes-in-keras/52173844#52173844
    y_true and y_pred are batches of N samples x N features
    the keras loss function archetype outputs the mean loss per sample
    i.e. N samples x 1d loss per sample
    e.g. MSE = K.mean(K.square(y_pred - y_true), axis=-1)
    the wrapping "weighted_masked_objective" function then returns the final batch-wise mean
    https://github.com/keras-team/keras/blob/2.2.4/keras/engine/training_utils.py#L374
    note that it is possible to perform all batchwise weighting inside the function...
    and to simply return a scalar (the wrapping function's mean operation then has no effect)...
    examples of implementation:
    A -> https://github.com/YannDubs/disentangling-vae
    rec loss: disentangling-vae uses sum of reconstruction array divided by batch size
    kl loss: takes mean kl per latent dimension, then sums
    B -> https://github.com/google-research/disentanglement_lib
    rec loss: takes sum of reconstruction per sample and then the mean
    kl loss: takes the sum of kl per sample, then the mean
    """

    def __init__(self, epochs, beta=1.0, capacity=0.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.epochs = epochs
        self.max_capacity = capacity
        self.capacity = tf.Variable(0.0, trainable=False)

    def call(self, inputs, **kwargs):
        Z_mu, Z_log_var = inputs
        # add loss and metrics
        # see page 5 in auto-encoding variational bayes paper
        kl = 1 + Z_log_var - K.square(Z_mu) - K.exp(Z_log_var)
        kl = K.sum(kl, axis=-1)  # sum across latents
        kl *= -0.5
        kl = K.mean(kl)  # take mean across batch
        kl_beta = self.beta * kl
        kl_cap = self.beta * K.abs(kl - self.capacity)
        return kl, kl_beta, kl_cap

    # callback for capacity update
    def capacity_update(self, epoch_step):
        # epochs seem to start at index 0
        if epoch_step == self.epochs - 1:
            new_capacity = self.max_capacity
        else:
            new_capacity = epoch_step / (self.epochs - 1) * self.max_capacity
        K.set_value(self.capacity, new_capacity)
        logger.info(f"updated capacity to {K.get_value(self.capacity)}")

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta": self.beta,
                "max_capacity": self.max_capacity,
                "capacity": 0.0,
                "capacityUpdate": self.capacity_update,
            }
        )
        return config


class VAE(models.Model):
    """ """

    def __init__(
        self,
        raw_dim: int,
        latent_dim: int,
        beta: float,
        capacity: float,
        epochs: int,
        model_key: str,
        seed=0,
        hidden_layer_dims: tuple = (32, 32, 32),
        activation_func: str = "swish",
        kernel_init: str = "glorot_uniform",
        **kwargs,
    ):
        tf.random.set_seed(seed)
        logger.info("initialising VAE...")
        super().__init__(**kwargs)
        self.model_key = model_key
        self.encoder = DeepLayer(
            name="DeepEncoder",
            hidden_layer_dims=hidden_layer_dims,
            activation_func=activation_func,
            kernel_init=kernel_init,
        )
        self.sampling = SamplingLayer(
            latent_dim,
            seed,
            initializers.TruncatedNormal(stddev=0.001, seed=seed),
            name="Sampling",
        )
        self.kl_divergence = KLDivergenceLayer(epochs, beta=beta, capacity=capacity, name="KLDivergence")
        self.decoder = DeepLayer(
            name="DeepDecoder",
            hidden_layer_dims=hidden_layer_dims,
            activation_func=activation_func,
            kernel_init=kernel_init,
        )
        self.x_hat = layers.Dense(raw_dim, activation="linear", name=f"OutputLayer")
        # setup metrics
        self.summary_metrics: dict[str, float] = {}

    def call(self, inputs, training=True, **kwargs):
        Z_mu, Z_log_var, Z = self.encode(inputs, training=training)
        kl, kl_beta, kl_cap = self.kl_divergence([Z_mu, Z_log_var])
        self.add_loss(kl_cap)
        X_hat = self.decode(Z, training=training)
        rec_loss = mse_sum(inputs, X_hat)
        self.add_loss(rec_loss)
        # update summary metrics
        for key, metric in zip(
            ["capacity_term", "kl", "kl_beta", "kl_beta_cap", "rec_loss"],
            [self.kl_divergence.capacity, kl, kl_beta, kl_cap, rec_loss],
        ):
            if key not in self.summary_metrics:
                self.summary_metrics[key] = metrics.Mean(name=key, dtype=np.float32)
            self.summary_metrics[key](metric)
        return X_hat

    def encode(self, X, training=False):
        X = self.encoder(X, training=training)
        Z_mu, Z_log_var, Z = self.sampling(X, training=training)
        return Z_mu, Z_log_var, Z

    def decode(self, Z, training=False):
        X = self.decoder(Z, training=training)
        X_hat = self.x_hat(X, training=training)
        return X_hat

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "theme": self.theme,
                "encoder": self.encoder,
                "sampling": self.sampling,
                "kl_divergence": self.kl_divergence,
                "decoder": self.decoder,
                "x_hat": self.x_hat,
                "encode": self.encode,
                "decode": self.decode,
            }
        )
        return config

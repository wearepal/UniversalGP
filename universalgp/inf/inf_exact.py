"""
Graph for exact inference.
"""

import numpy as np
import tensorflow as tf
from tensorflow import linalg as tfl
from tensorflow import math as tfm
from .. import util
from .base import Inference, VariableStore

# A jitter (diagonal) has to be added in the training covariance matrix in order to make Cholesky
# decomposition successfully
JITTER = 1e-2


class Store(VariableStore):
    """Stores the variables for the exact inference"""
    def build(self, input_shape):
        input_dim = int(input_shape[1])
        self.train_inputs = self.add_variable('train_inputs', [self.num_train, input_dim],
                                              trainable=False)
        self.train_outputs = self.add_variable('train_outputs', [self.num_train, self.output_dim],
                                               trainable=False)
        super().build(input_shape)

    def call(self, inputs):
        return self.train_inputs, self.train_outputs


class Exact(Inference):
    """Class for exact inference."""
    def __init__(self, args, lik_name, output_dim, num_train, inducing_inputs):
        super().__init__(args, num_train)
        self.store = Store(args, output_dim, num_train, inducing_inputs)
        self.lik, self.cov = util.construct_lik_and_cov(self, args, lik_name, output_dim)

    def inference(self, features, outputs, is_train):
        """Build graph for computing predictive mean and variance and log marginal likelihood.

        Args:
            train_inputs: inputs
            train_outputs: targets
            is_train: whether we're training
        Returns:
            negative log marginal likelihood
        """
        inputs = features['input']
        if is_train:
            # During training, we have to store the training data to compute predictions later on
            train_inputs, train_outputs = self.store(inputs)
            train_inputs.assign(inputs)
            train_outputs.assign(outputs)

        chol, alpha = self._build_interim_vals(inputs, outputs)
        # log marginal likelihood
        lml = self._build_log_marginal_likelihood(outputs, chol, alpha)

        return {'loss': -lml, 'LML': lml}

    def prediction(self, test_inputs):
        """Build graph for computing predictive mean and variance

        Args:
            test_inputs: test inputs
        Returns:
            predictive mean and variance
        """
        return self.apply(test_inputs['input'])

    def _apply(self, inputs):
        train_inputs, train_outputs = self.store(inputs)
        chol, alpha = self._build_interim_vals(train_inputs, train_outputs)

        # kxx_star (num_latent, num_train, num_test)
        kxx_star = self.cov[0](train_inputs, inputs)
        # f_star_mean (num_latent, num_test, 1)
        f_star_mean = tf.matmul(kxx_star, alpha, transpose_a=True)
        # Kx_star_x_star (num_latent, num_test)
        kx_star_x_star = self.cov[0](inputs)
        # v (num_latent, num_train, num_test)
        # v = tf.matmul(tf.matrix_inverse(chol), kxx_star)
        v = tfl.triangular_solve(chol, kxx_star)
        # var_f_star (same shape as Kx_star_x_star)
        var_f_star = tfl.tensor_diag_part(kx_star_x_star - tf.reduce_sum(v ** 2, axis=-2))
        pred_means, pred_vars = self.lik(tf.squeeze(f_star_mean, -1), variances=var_f_star)

        return pred_means[:, tf.newaxis], pred_vars[:, tf.newaxis]

    def _build_interim_vals(self, train_inputs, train_outputs):
        _, var = self.lik(0, variances=0)
        # kxx (num_train, num_train)
        kxx = self.cov[0](train_inputs) + var * tf.eye(tf.shape(input=train_inputs)[-2])

        jitter = JITTER * tf.eye(tf.shape(input=train_inputs)[-2])
        # chol (same size as kxx), add jitter has to be added
        chol = tfl.cholesky(kxx + jitter)
        # alpha = chol.T \ (chol \ train_outputs)
        alpha = tfl.cholesky_solve(chol, train_outputs)
        return chol, alpha

    @staticmethod
    def _build_log_marginal_likelihood(train_outputs, chol, alpha):

        # contract the batch dimension, quad_form (num_latent,)
        quad_form = tf.matmul(train_outputs, alpha, transpose_a=True)
        log_trace = util.log_cholesky_det(chol)
        #   tf.reduce_sum(tfm.log(tf.matrix_diag_part(chol)), -1)
        # log_marginal_likelihood (num_latent,)
        num_train = tf.cast(tf.shape(input=chol)[-1], dtype=tf.float32)
        log_marginal_likelihood = -0.5 * (quad_form + log_trace + num_train * tfm.log(np.pi))
        # Sum over num_latent in the end to get a scalar, this corresponds to mutliplying the
        # marginal likelihoods of all the latent functions
        return tf.reduce_sum(input_tensor=log_marginal_likelihood)

"""
Leave-one-out inference for Gaussian process models

Reference:
Carl Edward Rasmussen and Christopher K. I. Williams
The MIT Press, 2006. ISBN 0-262-18253-X. p116.
"""

import numpy as np
import tensorflow as tf
from .. import lik
from .. import util

JITTER = 1e-2


class Loo:
    """Class for inference based on LOO."""

    def __init__(self, cov_func, lik_func, num_train, *_):
        self.cov = cov_func
        self.lik = lik_func
        self.sn = self.lik.get_params()[0]
        with tf.variable_scope(None, "loo_inference"):
            self.train_inputs = tf.get_variable('train_inputs', [num_train, self.cov[0].input_dim], trainable=False)
            self.train_outputs = tf.get_variable('train_outputs', [num_train, len(self.cov)], trainable=False)

    def inference(self, features, outputs, is_train):
        """Build graph for computing predictive mean and variance and negative log probability.

        Args:
            train_inputs: inputs
            train_outputs: targets
            is_train: whether we're training
        Returns:
            negative log marginal likelihood
        """
        inputs = features['input']
        if is_train:
            # During training, we have to store the training data for computing the predictions later on
            inputs = self.train_inputs.assign(inputs)
            outputs = self.train_outputs.assign(outputs)

        chol, alpha = self._build_interim_vals(inputs, outputs)
        # precision = inv(kxx)
        precision = tf.cholesky_solve(chol, tf.eye(tf.shape(inputs)[-2]))
        precision_diag = tf.matrix_diag_part(precision)

        loo_fmu = outputs - alpha / precision_diag   # GMPL book eq. 5.12
        loo_fs2 = 1.0 / precision_diag               # GMPL book eq. 5.12

        # negative log probability (nlp), also called log pseudo-likelihood)
        nlp = - self._build_loo(outputs, loo_fmu, loo_fs2)

        return {'NLP': nlp}, []

    def predict(self, test_inputs):
        """Build graph for computing predictive mean and variance

        Args:
            test_inputs: test inputs
        Returns:
            predictive mean and variance
        """
        chol, alpha = self._build_interim_vals(self.train_inputs, self.train_outputs)

        # kxx_star (num_latent, num_train, num_test)
        kxx_star = self.cov[0].cov_func(self.train_inputs, test_inputs['input'])
        # f_star_mean (num_latent, num_test, 1)
        f_star_mean = tf.matmul(kxx_star, alpha, transpose_a=True)
        # Kx_star_x_star (num_latent, num_test)
        kx_star_x_star = self.cov[0].cov_func(test_inputs['input'])
        # v (num_latent, num_train, num_test)
        # v = tf.matmul(tf.matrix_inverse(chol), kxx_star)
        v = tf.matrix_triangular_solve(chol, kxx_star)
        # var_f_star (same shape as Kx_star_x_star)
        var_f_star = tf.diag_part(kx_star_x_star - tf.reduce_sum(v ** 2, -2))
        pred_means, pred_vars = self.lik.predict(tf.squeeze(f_star_mean, -1), var_f_star)

        return pred_means, pred_vars

    def _build_interim_vals(self, train_inputs, train_outputs):
        # kxx (num_train, num_train)
        kxx = self.cov[0].cov_func(train_inputs) + self.sn ** 2 * tf.eye(tf.shape(train_inputs)[-2])

        jitter = JITTER * tf.eye(tf.shape(train_inputs)[-2])
        # chol (same size as kxx), add jitter has to be added
        chol = tf.cholesky(kxx + jitter)
        # alpha = chol.T \ (chol \ train_outputs)
        alpha = tf.cholesky_solve(chol, train_outputs)
        return chol, alpha

    def _build_loo(self, train_outputs, loo_fmu, loo_fs2):
        pred_log_probability = self.lik.pred_log_prob(train_outputs, loo_fmu, loo_fs2)

        return tf.reduce_sum(pred_log_probability)

    def get_all_variables(self):
        """Returns all variables, not just the ones that are trained."""
        return [self.train_inputs, self.train_outputs]

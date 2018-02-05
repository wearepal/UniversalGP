"""
Graph for exact inference.
"""

import numpy as np
import tensorflow as tf
from .. import util

# A jitter (diagonal) has to be added in the training covariance matrix in order to make Cholesky decomposition
# successfully
JITTER = 1e-2


class Exact:
    def __init__(self, cov_func, lik_func):
        self.cov = cov_func
        self.lik = lik_func
        self.sn = self.lik.get_params()[0]

    def exact_inference(self, train_inputs, train_outputs, num_train, test_inputs):
        """Build graph for computing predictive mean and variance and negative log marginal likelihood.

        Args:
            train_inputs: inputs
            train_outputs: targets
            test_inputs: test inputs
            num_train: the number of trainings
        Returns:
            negative log marginal likelihood and predictive mean and variance
        """

        # kxx (num_train, num_train)
        kxx = self.cov.cov_func(train_inputs)[0] + self.sn ** 2 * tf.eye(tf.shape(train_inputs)[-2])
        jitter = JITTER * tf.eye(tf.shape(train_inputs)[-2])
        # chol (same size as kxx), add jitter has to be added
        chol = tf.cholesky(kxx + jitter)
        # alpha = chol.T \ (chol \ train_outputs)
        alpha = tf.cholesky_solve(chol, train_outputs)
        # negative log marginal likelihood
        nlml = - self._build_log_marginal_likelihood(train_outputs, chol, alpha, num_train)
        predictions = self._build_predict(train_inputs, test_inputs, chol, alpha)

        return nlml, predictions

    def _build_predict(self, train_inputs, test_inputs, chol, alpha):

        # kxx_star (num_latent, num_train, num_test)
        kxx_star = self.cov.cov_func(train_inputs, test_inputs)[0]
        # f_star_mean (num_latent, num_test, 1)
        f_star_mean = tf.matmul(kxx_star, alpha, transpose_a=True)
        # Kx_star_x_star (num_latent, num_test)
        kx_star_x_star = self.cov.cov_func(test_inputs)[0]
        # v (num_latent, num_train, num_test)
        v = tf.matmul(tf.matrix_inverse(chol), kxx_star)
        # var_f_star (same shape as Kx_star_x_star)
        var_f_star = tf.diag_part(kx_star_x_star - tf.reduce_sum(v ** 2, -2))
        pred_means, pred_vars = self.lik.predict(tf.squeeze(f_star_mean, -1), var_f_star)

        return pred_means, pred_vars

    @staticmethod
    def _build_log_marginal_likelihood(train_outputs, chol, alpha, num_train):

        # contract the batch dimension, quad_form (num_latent,)
        quad_form = tf.matmul(train_outputs, alpha, transpose_a=True)
        log_trace = util.log_cholesky_det(chol)
        #   tf.reduce_sum(tf.log(tf.matrix_diag_part(chol)), -1)
        # log_marginal_likelihood (num_latent,)
        log_marginal_likelihood = -0.5 * quad_form - 0.5 * log_trace - 0.5 * num_train * tf.log(np.pi)
        # sum over num_latent in the end to get a scalar, this corresponds to mutliplying the marginal likelihoods
        # of all the latent functions
        return tf.reduce_sum(log_marginal_likelihood)

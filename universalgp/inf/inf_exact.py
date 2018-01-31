"""
Graph for exact inference.
"""

import numpy as np
import tensorflow as tf


class Exact:
    def __init__(self, cov_func, lik_func):
        self.cov = cov_func
        self.lik = lik_func

        self.num_latent = len(self.cov)

    def inference(self, X, y, X_star):
        """Build graph for computing predictive mean and variance and negative log marginal likelihood.

        Args:
            X: inputs
            y: targets
            X_star: test inputs
        Returns:
            negative log marginal likelihood and predictive mean and variance
        """
        num_train = tf.shape(y)[0]
        # Kxx (num_latent, num_train, num_train)
        Kxx = [self.cov[i].cov_func(X[i, :, :])  # + std_dev**2 * tf.eye(num_train)
               for i in range(self.num_latent)]
        # L (same size as Kxx)
        L = tf.stack([tf.cholesky(kxx) for kxx in Kxx], 0)
        # alpha = L.T \ (L \ y)
        # b = L \ y means L @ b = y
        # α_interim (num_latent, num_train, 1)
        α_interim = tf.stack([tf.cholesky_solve(L[i, :, :], y[:, i, tf.newaxis]) for i in range(self.num_latent)], 0)
        # alpha (num_latent, num_train, 1)
        α = tf.cholesky_solve(tf.matrix_transpose(L), α_interim)
        return -self._build_log_marginal_likelihood(y, L, α), self._build_prediction(X, X_star, L, α)

    def _build_prediction(self, X, X_star, L, alpha):
        # Kxx_star (num_latent, num_train, num_test)
        Kxx_star = tf.stack([self.cov[i].cov_func(X[i, :, :], X_star) for i in range(self.num_latent)], 0)
        # f_star_mean (num_latent, num_test, 1)
        f_star_mean = tf.matmul(Kxx_star, alpha, transpose_a=True)
        # Kx_star_x_star (num_latent, num_test)
        Kx_star_x_star_diag = tf.stack([self.cov[i].diag_cov_func(X_star) for i in range(self.num_latent)], 0)
        # v (num_latent, num_train, num_test)
        v = tf.cholesky_solve(L, Kxx_star)
        # var_f_star (same shape as Kx_star_x_star)
        var_f_star = Kx_star_x_star_diag - tf.reduce_sum(v**2, -2)
        return tf.transpose(tf.squeeze(f_star_mean, -1)), tf.transpose(var_f_star)

    def _build_log_marginal_likelihood(self, y, L, α):
        num_train = tf.to_float(tf.shape(y)[0])
        # contract the batch dimension, quad_form (num_latent,)
        quad_form = tf.einsum('bl,lb->l', y, tf.squeeze(α, 2))
        log_trace = tf.reduce_sum(tf.log(tf.matrix_diag_part(L)), -1)
        # log_marginal_likelihood (num_latent,)
        log_marginal_likelihood = -0.5 * quad_form - log_trace - 0.5 * num_train * tf.log(np.pi)
        # sum over num_latent in the end to get a scalar, this corresponds to mutliplying the marginal likelihoods
        # of all the latent functions
        return tf.reduce_sum(log_marginal_likelihood)

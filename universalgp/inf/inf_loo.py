#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 13:56:02 2018

@author: zc223

Usage: make leave-one-out inference for Gaussian process models

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
    def __init__(self, cov_func, lik_func):
        self.cov = cov_func
        self.lik = lik_func
        self.sn = self.lik.get_params()[0]

    def inference(self, train_inputs, train_outputs, *_):
        """Build graph for computing negative log probability.

        Args:
            train_inputs: inputs
            train_outputs: targets
        Returns:
            negative log probability
        """

        self.train_inputs = train_inputs
        # kxx (num_train, num_train)
        kxx = self.cov[0].cov_func(train_inputs) + self.sn ** 2 * tf.eye(tf.shape(train_inputs)[-2])

        jitter = JITTER * tf.eye(tf.shape(train_inputs)[-2])
        # chol (same size as kxx), add jitter has to be added
        self.chol = tf.cholesky(kxx + jitter)
        # alpha = chol.T \ (chol \ train_outputs)
        self.alpha = tf.cholesky_solve(self.chol, train_outputs)
        # precision = inv(kxx)
        precision = tf.cholesky_solve(self.chol, tf.eye(tf.shape(train_inputs)[-2]))
        precision_diag = tf.matrix_diag_part(precision)

        loo_fmu = train_outputs - self.alpha / precision_diag   # GMPL book eq. 5.12
        loo_fs2 = 1.0 / precision_diag                          # GMPL book eq. 5.12

        # negative log probability (nlp), also called log pseudo-likelihood)
        nlp = - self._build_loo(train_outputs, loo_fmu, loo_fs2)
        return {'NLP': nlp}, []

    def predict(self, test_inputs):
        """Build graph for computing predictive mean and variance

        Args:
            test_inputs: test inputs
        Returns:
            predictive mean and variance
        """
        # kxx_star (num_latent, num_train, num_test)
        kxx_star = self.cov[0].cov_func(self.train_inputs, test_inputs)
        # f_star_mean (num_latent, num_test, 1)
        f_star_mean = tf.matmul(kxx_star, self.alpha, transpose_a=True)
        # Kx_star_x_star (num_latent, num_test)
        kx_star_x_star = self.cov[0].cov_func(test_inputs)
        # v (num_latent, num_train, num_test)
        # v = tf.matmul(tf.matrix_inverse(chol), kxx_star)
        v = tf.matrix_triangular_solve(self.chol, kxx_star)
        # var_f_star (same shape as Kx_star_x_star)
        var_f_star = tf.diag_part(kx_star_x_star - tf.reduce_sum(v ** 2, -2))
        pred_means, pred_vars = self.lik.predict(tf.squeeze(f_star_mean, -1), var_f_star)

        return pred_means, pred_vars

    def _build_loo(self, train_outputs, loo_fmu, loo_fs2):
        pred_log_probability = self.lik.pred_log_prob(train_outputs, loo_fmu, loo_fs2)

        return tf.reduce_sum(pred_log_probability)

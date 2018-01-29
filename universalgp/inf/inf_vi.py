#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 12:41:54 2018

@author: zc223
"""

from .. import mean
from .. import cov
from .. import lik
from .. import util


class VariationalInference:

    def __init__(self,
                 inducing_inputs,
                 # mean_func=mean.ZeroOffset,
                 cov_func=cov.SquaredExponential,
                 lik_func=lik.LikelihoodGaussian,
                 num_components=1,
                 num_samples=100):
        # self.mean = mean_func
        self.cov = cov_func
        self.lik = lik_func

        self.num_components = num_components
        self.num_latent = len(self.cov)
        self.num_samples = num_samples

        self.num_inducing = inducing_inputs.shape[1]
        self.input_dim = inducing_inputs.shape[2]

    def _build_entropy(self, weights, means, covars):
        # First build half a square matrix of normals. This avoids re-computing symmetric normals.
        log_normal_probs = util.init_list(0.0, [self.num_components, self.num_components])
        for i in range(self.num_components):
            for j in range(i, self.num_components):
                for k in range(self.num_latent):
                    if self.diag_post:
                        normal = util.DiagNormal(means[i, k, :], covars[i, k, :] +
                                                                 covars[j, k, :])
                    else:
                        if i == j:
                            # Compute chol(2S) = sqrt(2)*chol(S).
                            covars_sum = tf.sqrt(2.0) * covars[i, k, :, :]
                        else:
                            # TODO(karl): Can we just stay in cholesky space somehow?
                            covars_sum = tf.cholesky(util.mat_square(covars[i, k, :, :]) +
                                                     util.mat_square(covars[j, k, :, :]))
                        normal = util.CholNormal(means[i, k, :], covars_sum)
                    log_normal_probs[i][j] += normal.log_prob(means[j, k, :])

        # Now compute the entropy.
        entropy = 0.0
        for i in range(self.num_components):
            weighted_log_probs = util.init_list(0.0, [self.num_components])
            for j in range(self.num_components):
                if i <= j:
                    weighted_log_probs[j] = tf.log(weights[j]) + log_normal_probs[i][j]
                else:
                    weighted_log_probs[j] = tf.log(weights[j]) + log_normal_probs[j][i]

            entropy -= weights[i] * util.logsumexp(tf.stack(weighted_log_probs))

        return entropy

    def _build_cross_ent(self, weights, means, covars, kernel_chol):
        cross_ent = 0.0
        for i in range(self.num_components):
            sum_val = 0.0
            for j in range(self.num_latent):
                if self.diag_post:
                    # TODO(karl): this is a bit inefficient since we're not making use of the fact
                    # that covars is diagonal. A solution most likely involves a custom tf op.
                    trace = tf.trace(tf.cholesky_solve(kernel_chol[j, :, :],
                                                         tf.diag(covars[i, j, :])))
                else:
                    trace = tf.reduce_sum(util.diag_mul(
                        tf.cholesky_solve(kernel_chol[j, :, :], covars[i, j, :, :]),
                        tf.transpose(covars[i, j, :, :])))

                sum_val += (util.CholNormal(means[i, j, :], kernel_chol[j, :, :]).log_prob(0.0) -
                            0.5 * trace)

            cross_ent += weights[i] * sum_val

        return cross_ent

    def _build_ell(self, weights, means, covars, inducing_inputs,
                   kernel_chol, train_inputs, train_outputs):
        kern_prods, kern_sums = self._build_interim_vals(kernel_chol, inducing_inputs, train_inputs)
        ell = 0
        for i in range(self.num_components):
            covar_input = covars[i, :, :] if self.diag_post else covars[i, :, :, :]
            latent_samples = self._build_samples(kern_prods, kern_sums,
                                                 means[i, :, :], covar_input)
            ell += weights[i] * tf.reduce_sum(self.likelihood.log_cond_prob(train_outputs,
                                                                            latent_samples))

        return ell / self.num_samples

    def _build_interim_vals(self, kernel_chol, inducing_inputs, train_inputs):
        kern_prods = util.init_list(0.0, [self.num_latent])
        kern_sums = util.init_list(0.0, [self.num_latent])
        for i in range(self.num_latent):
            ind_train_kern = self.kernels[i].kernel(inducing_inputs[i, :, :], train_inputs)
            # Compute A = Kxz.Kzz^(-1) = (Kzz^(-1).Kzx)^T.
            kern_prods[i] = tf.transpose(tf.cholesky_solve(kernel_chol[i, :, :], ind_train_kern))
            # We only need the diagonal components.
            kern_sums[i] = (self.kernels[i].diag_kernel(train_inputs) -
                            util.diag_mul(kern_prods[i], ind_train_kern))

        kern_prods = tf.stack(kern_prods, 0)
        kern_sums = tf.stack(kern_sums, 0)
        return kern_prods, kern_sums




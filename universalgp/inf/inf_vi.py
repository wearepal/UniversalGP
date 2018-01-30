#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 12:41:54 2018

@author: zc223
"""
import tensorflow as tf
from .. import mean
from .. import cov
from .. import lik
from .. import util


class Variational:

    def __init__(self, cov_func, lik_func, diag_post=False, num_components=1, num_samples=100):

        # self.mean = mean_func
        self.cov = cov_func
        self.lik = lik_func

        self.num_components = num_components
        self.num_latent = len(self.cov)
        # Save whether our posterior is diagonal or not.
        self.diag_post = diag_post
        self.num_samples = num_samples

    def variation_inference(self,
                            raw_weights,
                            raw_means,
                            raw_covars,
                            raw_inducing_inputs,
                            train_inputs,
                            train_outputs,
                            num_train,
                            test_inputs):

        # First transform all raw variables into their internal form.
        # Use softmax(raw_weights) to keep all weights normalized.
        weights = tf.exp(raw_weights) / tf.reduce_sum(tf.exp(raw_weights))

        if self.diag_post:
            # Use exp(raw_covars) so as to guarantee the diagonal matrix remains positive definite.
            covars = tf.exp(raw_covars)
        else:
            # Use vec_to_tri(raw_covars) so as to only optimize over the lower triangular portion.
            # We note that we will always operate over the cholesky space internally.
            covars_list = [None] * self.num_components
            for i in range(self.num_components):
                mat = util.vec_to_tri(raw_covars[i, :, :])
                diag_mat = tf.matrix_diag(tf.matrix_diag_part(mat))
                exp_diag_mat = tf.matrix_diag(tf.exp(tf.matrix_diag_part(mat)))
                covars_list[i] = mat - diag_mat + exp_diag_mat
            covars = tf.stack(covars_list, 0)
        # Both inducing inputs and the posterior means can vary freely so don't change them.
        means = raw_means
        inducing_inputs = raw_inducing_inputs

        # Build the matrices of covariances between inducing inputs.
        kernel_mat = [self.cov[i].cov_func(inducing_inputs[i, :, :])
                      for i in range(self.num_latent)]
        kernel_chol = tf.stack([tf.cholesky(k) for k in kernel_mat], 0)


        # Now build the objective function.
        entropy = self._build_entropy(weights, means, covars)
        cross_ent = self._build_cross_ent(weights, means, covars, kernel_chol)
        ell = self._build_ell(weights, means, covars, inducing_inputs,
                              kernel_chol, train_inputs, train_outputs)
        batch_size = tf.to_float(tf.shape(train_inputs)[0])
        nelbo = -((batch_size / num_train) * (entropy + cross_ent) + ell)

        # Finally, build the prediction function.
        predictions = self._build_predict(weights, means, covars, inducing_inputs,
                                          kernel_chol, test_inputs)
        return nelbo, predictions

    def _build_predict(self, weights, means, covars, inducing_inputs,
                       kernel_chol, test_inputs):
        kern_prods, kern_sums = self._build_interim_vals(kernel_chol, inducing_inputs, test_inputs)
        pred_means = util.init_list(0.0, [self.num_components])
        pred_vars = util.init_list(0.0, [self.num_components])
        for i in range(self.num_components):
            covar_input = covars[i, :, :] if self.diag_post else covars[i, :, :, :]
            sample_means, sample_vars = self._build_sample_info(kern_prods, kern_sums,
                                                                means[i, :, :], covar_input)
            pred_means[i], pred_vars[i] = self.lik.predict(sample_means, sample_vars)

        pred_means = tf.stack(pred_means, 0)
        pred_vars = tf.stack(pred_vars, 0)

        # Compute the mean and variance of the gaussian mixture from their components.
        # weights = tf.expand_dims(tf.expand_dims(weights, 1), 1)
        weights = weights[:, tf.newaxis, tf.newaxis]
        weighted_means = tf.reduce_sum(weights * pred_means, 0)
        weighted_vars = (tf.reduce_sum(weights * (pred_means ** 2 + pred_vars), 0) -
                         tf.reduce_sum(weights * pred_means, 0) ** 2)
        return weighted_means, weighted_vars

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
            ell += weights[i] * tf.reduce_sum(self.lik.log_cond_prob(train_outputs, latent_samples))

        return ell / self.num_samples

    def _build_interim_vals(self, kernel_chol, inducing_inputs, train_inputs):
        kern_prods = util.init_list(0.0, [self.num_latent])
        kern_sums = util.init_list(0.0, [self.num_latent])
        for i in range(self.num_latent):
            ind_train_kern = self.cov[i].cov_func(inducing_inputs[i, :, :], train_inputs)
            # Compute A = Kxz.Kzz^(-1) = (Kzz^(-1).Kzx)^T.
            kern_prods[i] = tf.transpose(tf.cholesky_solve(kernel_chol[i, :, :], ind_train_kern))
            # We only need the diagonal components.
            kern_sums[i] = (self.cov[i].diag_cov_func(train_inputs) -
                            util.diag_mul(kern_prods[i], ind_train_kern))

        kern_prods = tf.stack(kern_prods, 0)
        kern_sums = tf.stack(kern_sums, 0)
        return kern_prods, kern_sums

    def _build_samples(self, kern_prods, kern_sums, means, covars):
        sample_means, sample_vars = self._build_sample_info(kern_prods, kern_sums, means, covars)
        batch_size = tf.shape(sample_means)[0]
        return (sample_means + tf.sqrt(sample_vars) *
                tf.random_normal([self.num_samples, batch_size, self.num_latent]))

    def _build_sample_info(self, kern_prods, kern_sums, means, covars):
        sample_means = util.init_list(0.0, [self.num_latent])
        sample_vars = util.init_list(0.0, [self.num_latent])
        for i in range(self.num_latent):
            if self.diag_post:
                quad_form = util.diag_mul(kern_prods[i, :, :] * covars[i, :],
                                          tf.transpose(kern_prods[i, :, :]))
            else:
                full_covar = covars[i, :, :] @ tf.transpose(covars[i, :, :])
                quad_form = util.diag_mul(kern_prods[i, :, :] @ full_covar,
                                          tf.transpose(kern_prods[i, :, :]))
            sample_means[i] = kern_prods[i, :, :] @ means[i, :, tf.newaxis]
            sample_vars[i] = (kern_sums[i, :] + quad_form)[:, tf.newaxis]

        sample_means = tf.concat(sample_means, 1)
        sample_vars = tf.concat(sample_vars, 1)
        return sample_means, sample_vars


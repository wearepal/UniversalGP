"""
Created on Mon Jan 29 12:41:54 2018

@author: zc223

Usage: make variational inference for generic Gaussian process models
"""
import tensorflow as tf
from .. import util

JITTER = 1e-2


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
        jitter = JITTER * tf.eye(tf.shape(inducing_inputs)[-2])

        kernel_chol = tf.stack([tf.cholesky(k + jitter) for k in kernel_mat], 0)

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

        return {'NELBO': tf.squeeze(nelbo)}, predictions

    def _build_predict(self, weights, means, covars, inducing_inputs, kernel_chol, test_inputs):
        """Construct predictive distribution

        Args:
            weights: (num_components,)
            means: shape: (num_components, num_latent, num_inducing)
            covars: shape: (num_components, num_latent, num_inducing[, num_inducing])
            inducing_inputs: (num_latent, num_inducing, input_dim)
            kernel_chol: (num_latent, num_inducing, num_inducing)
            test_inputs: (batch_size, input_dim)
        Returns:
            means and variances of the predictive distribution
        """
        kern_prods, kern_sums = self._build_interim_vals(kernel_chol, inducing_inputs, test_inputs)
        sample_means, sample_vars = self._build_sample_info(kern_prods, kern_sums, means, covars)
        pred_means, pred_vars = self.lik.predict(sample_means, sample_vars)

        # Compute the mean and variance of the gaussian mixture from their components.
        # weights = tf.expand_dims(tf.expand_dims(weights, 1), 1)
        weights = weights[:, tf.newaxis, tf.newaxis]
        weighted_means = tf.reduce_sum(weights * pred_means, 0)
        weighted_vars = (tf.reduce_sum(weights * (pred_means ** 2 + pred_vars), 0) -
                         tf.reduce_sum(weights * pred_means, 0) ** 2)
        return tf.squeeze(weighted_means), tf.squeeze(weighted_vars)

    def _build_entropy(self, weights, means, covars):
        """Construct entropy.

        Args:
            weights: shape: (num_components)
            means: shape: (num_components, num_latent, num_inducing)
            covars: shape: (num_components, num_latent, num_inducing[, num_inducing])
        Returns:
            Entropy (scalar)
        """
        # First build a square matrix of normals.
        if self.diag_post:
            # construct normal distributions for all combinations of compontents
            normal = util.DiagNormal(means, covars[tf.newaxis, ...] + covars[:, tf.newaxis, ...])
        else:
            # TODO(karl): Can we just stay in cholesky space somehow?
            square = util.mat_square(covars)
            covars_sum = tf.cholesky(square[tf.newaxis, ...] + square[:, tf.newaxis, ...])
            normal = util.CholNormal(means, covars_sum)
        # compute log probability of all means in all normal distributions
        # then sum over all latent functions
        # shape of log_normal_probs: (num_components, num_components)
        log_normal_probs = tf.reduce_sum(normal.log_prob(means[:, tf.newaxis, ...]), axis=-1)

        # Now compute the entropy.
        # broadcast `weights` into dimension 1, then do `logsumexp` in that dimension
        weighted_logsumexp_probs = tf.reduce_logsumexp(tf.log(weights) + log_normal_probs, 1)
        # multiply with weights again and then sum over it all
        return -tf.tensordot(weights, weighted_logsumexp_probs, 1)

    def _build_cross_ent(self, weights, means, covars, kernel_chol):
        """Construct the cross-entropy.

        Args:
            weights: shape: (num_components)
            means: shape: (num_components, num_latent, num_inducing)
            covars: shape: (num_components, num_latent, num_inducing[, num_inducing])
            kernel_chol: shape: (num_latent, num_inducing, num_inducing)
        Returns:
            Cross entropy as scalar
        """
        if self.diag_post:
            # TODO(karl): this is a bit inefficient since we're not making use of the fact
            # that covars is diagonal. A solution most likely involves a custom tf op.

            # shape of trace: (num_components, num_latent)
            trace = tf.trace(util.cholesky_solve_br(kernel_chol, tf.matrix_diag(covars)))
        else:
            trace = tf.reduce_sum(util.diag_mul(util.cholesky_solve_br(kernel_chol, covars),
                                                tf.matrix_transpose(covars)), axis=-1)

        # sum_val has the same shape as weights
        sum_val = tf.reduce_sum(util.CholNormal(means, kernel_chol).log_prob(0.0) - 0.5 * trace, -1)

        # dot product of weights and sum_val
        cross_ent = tf.tensordot(weights, sum_val, 1)

        return cross_ent

    def _build_ell(self, weights, means, covars, inducing_inputs, kernel_chol, train_inputs, train_outputs):
        """Construct the Expected Log Likelihood

        Args:
            weights: (num_components,)
            means: shape: (num_components, num_latent, num_inducing)
            covars: shape: (num_components, num_latent, num_inducing[, num_inducing])
            inducing_inputs: (num_latent, num_inducing, input_dim)
            kernel_chol: (num_latent, num_inducing, num_inducing)
            train_inputs: (batch_size, input_dim)
            train_outputs: (batch_size, num_latent)
        Returns:
            Expected log likelihood as scalar
        """
        kern_prods, kern_sums = self._build_interim_vals(kernel_chol, inducing_inputs, train_inputs)
        # shape of `latent_samples`: (num_components, num_samples, batch_size, num_latent)
        latent_samples = self._build_samples(kern_prods, kern_sums, means, covars)
        ell_by_compontent = tf.reduce_sum(self.lik.log_cond_prob(train_outputs, latent_samples), axis=[1, 2])

        # dot product
        ell = tf.tensordot(weights, ell_by_compontent, 1)
        return ell / self.num_samples

    def _build_interim_vals(self, kernel_chol, inducing_inputs, train_inputs):
        """Helper function for `_build_ell`

        Args:
            kernel_chol: Tensor(num_latent, num_inducing, num_inducing)
            inducing_inputs: Tensor(num_latent, num_inducing, input_dim)
            train_inputs: Tensor(batch_size, input_dim)
        Returns:
            `kern_prods` (num_latent, batch_size, num_inducing) and `kern_sums` (num_latent, batch_size)
        """
        # shape of ind_train_kern: (num_latent, num_inducing, batch_size)

        kern_prods = [0.0 for _ in range(self.num_latent)]
        kern_sums = [0.0 for _ in range(self.num_latent)]

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
        """Produce samples according to the given distribution.

        Args:
            kern_prods: (num_latent, batch_size, num_inducing)
            kern_sums: (num_latent, batch_size)
            means: (num_components, num_latent, num_inducing)
            covars: (num_components, num_latent, num_inducing[, num_inducing])
        Returns:
        """
        sample_means, sample_vars = self._build_sample_info(kern_prods, kern_sums, means, covars)
        batch_size = tf.shape(sample_means)[-2]
        return (sample_means[:, tf.newaxis, ...] + tf.sqrt(sample_vars)[:, tf.newaxis, ...] *
                tf.random_normal([self.num_components, self.num_samples, batch_size, self.num_latent]))

    def _build_sample_info(self, kern_prods, kern_sums, means, covars):
        """Get means and variances of a distribution

        Args:
            kern_prods: (num_latent, batch_size, num_inducing)
            kern_sums: (num_latent, batch_size)
            means: (num_components, num_latent, num_inducing)
            covars: (num_components, num_latent, num_inducing[, num_inducing])
        Returns:
            sample_means (num_components, batch_size, num_latent), sample_vars (num_components, batch_size, num_latent)
        """
        if self.diag_post:
            quad_form = util.diag_mul(kern_prods * covars[..., tf.newaxis, :], tf.matrix_transpose(kern_prods))
        else:
            full_covar = util.mat_square(covars)  # same shape as covars
            quad_form = util.diag_mul(util.matmul_br(kern_prods, full_covar), tf.matrix_transpose(kern_prods))
        sample_means = util.matmul_br(kern_prods, means[..., tf.newaxis])  # (num_components, num_latent, batch_size, 1)
        sample_vars = tf.matrix_transpose(kern_sums + quad_form)  # (num_components, x, num_latent)
        return tf.matrix_transpose(tf.squeeze(sample_means, -1)), sample_vars

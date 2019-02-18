"""
Variational inference for generic Gaussian process models
"""
import tensorflow as tf
from tensorflow import math as tfm
from tensorflow_probability import distributions as tfd
import numpy as np
from .. import util
from .base import Inference

JITTER = 1e-2

tf.compat.v1.app.flags.DEFINE_integer('num_components', 1,
                                      'Number of mixture of Gaussians components')
tf.compat.v1.app.flags.DEFINE_integer(
    'num_samples', 100, 'Number of samples for mean and variance estimate of likelihood')
tf.compat.v1.app.flags.DEFINE_boolean('diag_post', False,
                                      'Whether the posterior is diagonal or not')
tf.compat.v1.app.flags.DEFINE_boolean('optimize_inducing', True,
                                      'Whether to optimize the inducing inputs in training')
tf.compat.v1.app.flags.DEFINE_boolean(
    'use_loo', False, 'Whether to use the LOO (leave one out) loss (for hyper parameters)')


class Variational(Inference):
    """
    Defines inference for Variational Inference
    """

    def build(self, input_shape):
        """Create a new variational inference object which will keep track of all variables."""
        input_dim = int(input_shape[1])
        self.lik, self.cov = util.construct_lik_and_cov(self, self.args, self.lik_name, input_dim,
                                                        self.output_dim)
        self.num_latents = self.output_dim

        # Initialize inducing inputs if they are provided
        if not hasattr(self, 'inducing_inputs_init'):
            # Only the number of inducing inputs is given -> just specify the shape
            inducing_params = {'shape': [self.num_latents, self.num_inducing, input_dim]}
        else:
            # Repeat the inducing inputs for all latent processes if we haven't been given
            # individually specified inputs per process.
            if self.inducing_inputs_init.ndim == 2:
                inducing_inputs = np.tile(self.inducing_inputs_init[np.newaxis, :, :],
                                          reps=[self.num_latents, 1, 1])
            # Initialize with the given values
            inducing_params = {
                'shape': inducing_inputs.shape,
                'initializer': tf.keras.initializers.Constant(inducing_inputs),
            }

        num_components = self.args['num_components']
        # Initialize all variables
        # Define all parameters that get optimized directly in raw form. Some parameters get
        # transformed internally to maintain certain pre-conditions.

        self.inducing_inputs = self.add_variable("inducing_inputs", **inducing_params,
                                                 trainable=self.args['optimize_inducing'],
                                                 dtype=tf.float32)

        zeros = tf.keras.initializers.Zeros()
        self.raw_weights = self.add_variable("raw_weights", [num_components], initializer=zeros,
                                             dtype=tf.float32)
        self.means = self.add_variable(
            "means", [num_components, self.num_latents, self.num_inducing], initializer=zeros,
            dtype=tf.float32)
        if self.args['diag_post']:
            self.raw_covars = self.add_variable(
                "raw_covars", [num_components, self.num_latents, self.num_inducing],
                initializer=tf.keras.initializers.Ones())
        else:
            self.raw_covars = self.add_variable(
                "raw_covars",
                shape=[num_components, self.num_latents] + util.tri_vec_shape(self.num_inducing),
                initializer=zeros)
        super().build(input_shape)

    def _transform_variables(self):
        """Transorm variables that were stored in a more compact form.

        Doing it like this allows us to put certain constraints on the variables.
        """
        # Use softmax(raw_weights) to keep all weights normalized.
        weights = tf.nn.softmax(self.raw_weights)

        if self.args['diag_post']:
            # Use softplus(raw_covars) to guarantee the diagonal matrix remains positive definite.
            chol_covars = tf.nn.softplus(self.raw_covars)
        else:
            # Use vec_to_tri(raw_covars) so as to only optimize over the lower triangular portion.
            # We note that we will always operate over the cholesky space internally.
            triangle = util.vec_to_tri(self.raw_covars)
            chol_covars = tfd.matrix_diag_transform(triangle, transform=tf.nn.softplus)

        # Build the matrices of covariances between inducing inputs.
        kernel_mat = tf.stack([self.cov[i](self.inducing_inputs[i, :, :])
                               for i in range(self.num_latents)], 0)
        jitter = JITTER * tf.eye(tf.shape(input=self.inducing_inputs)[-2])

        kernel_chol = tf.linalg.cholesky(kernel_mat + jitter)
        return weights, chol_covars, kernel_chol

    def inference(self, features, outputs, is_train):
        """Build graph for computing negative evidence lower bound and predictive mean and variance

        Args:
            train_inputs: inputs
            train_outputs: targets
        Returns:
            negative evidence lower bound and variables to train
        """
        # First transform all raw variables into their internal form.
        weights, chol_covars, kernel_chol = self._transform_variables()

        # Build the objective function.
        entropy = self._build_entropy(weights, self.means, chol_covars)
        cross_ent = self._build_cross_ent(weights, self.means, chol_covars, kernel_chol)
        ell = self._build_ell(weights, self.means, chol_covars, self.inducing_inputs, kernel_chol,
                              features, outputs, is_train)
        batch_size = tf.cast(tf.shape(input=outputs)[0], dtype=tf.float32)
        nelbo = -((batch_size / self.num_train) * (entropy + cross_ent) + ell)

        obj_funcs = dict(elbo=-nelbo, entropy=(batch_size / self.num_train) * entropy,
                         cross_ent=(batch_size / self.num_train) * cross_ent, ell=ell)
        if self.args['use_loo']:
            # Compute LOO loss only when necessary
            loo_loss = self._build_loo_loss(weights, self.means, chol_covars, self.inducing_inputs,
                                            kernel_chol, features, outputs)
            return {**obj_funcs, 'NELBO': tf.squeeze(nelbo), 'LOO_VARIATIONAL': loo_loss,
                    'loss': tf.squeeze(nelbo) + loo_loss}
        return {**obj_funcs, 'loss': tf.squeeze(nelbo)}

    def prediction(self, test_inputs):
        """Make predictions"""
        return self.apply(test_inputs['input'])

    def _apply(self, inputs):
        """Construct predictive distribution

        weights: (num_components,)
        means: shape: (num_components, num_latents, num_inducing)
        chol_covars: shape: (num_components, num_latents, num_inducing[, num_inducing])
        inducing_inputs: (num_latents, num_inducing, input_dim)
        kernel_chol: (num_latents, num_inducing, num_inducing)

        Args:
            inputs: (batch_size, input_dim)
        Returns:
            means and variances of the predictive distribution
        """
        # Transform all raw variables into their internal form.
        weights, chol_covars, kernel_chol = self._transform_variables()

        kern_prods, kern_sums = self._build_interim_vals(kernel_chol, self.inducing_inputs, inputs)
        sample_means, sample_vars = self._build_sample_info(kern_prods, kern_sums, self.means,
                                                            chol_covars)
        pred_means, pred_vars = self.lik.predict(sample_means, sample_vars)

        # Compute the mean and variance of the gaussian mixture from their components.
        # weights = tf.expand_dims(tf.expand_dims(weights, 1), 1)
        weights = weights[:, tf.newaxis, tf.newaxis]
        weighted_means = tf.reduce_sum(input_tensor=weights * pred_means, axis=0)
        weighted_vars = (tf.reduce_sum(input_tensor=weights * (pred_means ** 2 + pred_vars), axis=0) -
                         tf.reduce_sum(input_tensor=weights * pred_means, axis=0) ** 2)
        return weighted_means, weighted_vars

    def _build_entropy(self, weights, means, chol_covars):
        """Construct entropy.

        Args:
            weights: shape: (num_components)
            means: shape: (num_components, num_latents, num_inducing)
            chol_covars: shape: (num_components, num_latents, num_inducing[, num_inducing])
        Returns:
            Entropy (scalar)
        """

        # This part is to compute the product of the pdf of normal distributions
        """
        chol_component_covar = []
        component_mean = []
        component_covar =[]
        covar_shape = tf.shape(chol_covars)[-2:]
        mean_shape = tf.shape(means)[-1:]

        # \Sigma_new = (\sum_{i=1}^{num_latents}( \Sigma_i^-1) )^{-1}
        # \Mu_new = \Sigma_new * (\sum_{i=1}^{num_latents} \Sigma_i^{-1} * \mu_i)
        for i in range(self.num_components):
            temp_cov = tf.zeros(covar_shape)
            temp_mean = tf.zeros(mean_shape)[..., tf.newaxis]

            for k in range(self.num_latents):
                # Compute the sum of (\Sigma_i)^{-1}
                temp_cov += tf.cholesky_solve(chol_covars[i, k, :, :], tf.eye(covar_shape[0]))
                # Compute the sum of (\Sigma_i)^{-1} * \mu_i
                temp_mean += tf.cholesky_solve(chol_covars[i, k, :, :],
                                               means[i, k, :, tf.newaxis])

            # Compute \Sigma_new = temp_cov^{-1}
            temp_chol_covar = tf.cholesky(temp_cov)
            temp_component_covar = tf.cholesky_solve(temp_chol_covar, tf.eye(covar_shape[0]))
            component_covar.append(temp_component_covar)
            # Compute \Mu_new = \Sigma_new * (\sum_{i=1}^{num_latents} \Sigma_i^{-1} * \mu_i)
            temp_component_mean = temp_component_covar @ temp_mean
            component_mean.append(temp_component_mean)

            # Some functions need cholesky of \Sigma_new
            chol_component_covar.append(tf.cholesky(temp_component_covar))

        chol_component_covar = tf.stack(chol_component_covar, 0)
        component_covar = tf.stack(component_covar, 0)
        component_mean = tf.squeeze(tf.stack(component_mean, 0), -1)
        """
        # First build a square matrix of normals.
        if self.args['diag_post']:
            # construct normal distributions for all combinations of components
            variational_dist = tfd.MultivariateNormalDiag(
                means, tf.sqrt(chol_covars[tf.newaxis, ...] + chol_covars[:, tf.newaxis, ...]))
        else:
            if self.args['num_components'] == 1:
                # Use the fact that chol(S + S) = sqrt(2) * chol(S)
                chol_covars_sum = tf.sqrt(2.) * chol_covars[tf.newaxis, ...]
            else:
                # Here we use the original component_covar directly
                # TODO: Can we just stay in cholesky space somehow?
                component_covar = util.mat_square(chol_covars)
                chol_covars_sum = tf.linalg.cholesky(component_covar[tf.newaxis, ...] +
                                              component_covar[:, tf.newaxis, ...])
            # The class MultivariateNormalTriL only accepts cholesky decompositions of covariances
            variational_dist = tfd.MultivariateNormalTriL(means[tf.newaxis, ...], chol_covars_sum)

        # compute log probability of all means in all normal distributions
        # then sum over all latent functions
        # shape of log_normal_probs: (num_components, num_components)
        log_normal_probs = tf.reduce_sum(input_tensor=variational_dist.log_prob(means[:, tf.newaxis, ...]), axis=-1)

        # Now compute the entropy.
        # broadcast `weights` into dimension 1, then do `logsumexp` in that dimension
        weighted_logsumexp_probs = tf.reduce_logsumexp(input_tensor=tfm.log(weights) + log_normal_probs, axis=1)
        # multiply with weights again and then sum over it all
        return -util.mul_sum(weights, weighted_logsumexp_probs)

    def _build_cross_ent(self, weights, means, chol_covars, kernel_chol):
        """Construct the cross-entropy.

        Args:
            weights: shape: (num_components)
            means: shape: (num_components, num_latents, num_inducing)
            chol_covars: shape: (num_components, num_latents, num_inducing[, num_inducing])
            kernel_chol: shape: (num_latents, num_inducing, num_inducing)
        Returns:
            Cross entropy as scalar
        """
        if self.args['diag_post']:
            # TODO(karl): this is a bit inefficient since we're not making use of the fact
            # that chol_covars is diagonal. A solution most likely involves a custom tf op.

            # shape of trace: (num_components, num_latents)
            trace = tf.linalg.trace(util.cholesky_solve_br(kernel_chol, tf.linalg.diag(chol_covars)))
        else:
            trace = tf.reduce_sum(input_tensor=util.mul_sum(util.cholesky_solve_br(kernel_chol, chol_covars),
                                               chol_covars), axis=-1)

        # sum_val has the same shape as weights
        gaussian = tfd.MultivariateNormalTriL(means, kernel_chol)
        sum_val = tf.reduce_sum(input_tensor=gaussian.log_prob([0.0]) - 0.5 * trace, axis=-1)

        # weighted sum of weights and sum_val
        cross_ent = util.mul_sum(weights, sum_val)

        return cross_ent

    def _build_loo_loss(self, weights, means, chol_covars, inducing_inputs, kernel_chol, features,
                        train_outputs):
        """Construct leave out one loss
        Args:
            weights: (num_components,)
            means: shape: (num_components, num_latent, num_inducing)
            chold_covars: shape: (num_components, num_latent, num_inducing[, num_inducing])
            inducing_inputs: (num_latent, num_inducing, input_dim)
            kernel_chol: (num_latent, num_inducing, num_inducing)
            train_inputs: (batch_size, input_dim)
            train_outputs: (batch_size, num_latent)
        Returns:
            LOO loss
        """
        kern_prods, kern_sums = self._build_interim_vals(kernel_chol, inducing_inputs,
                                                         features['input'])
        loss = 0
        latent_samples = self._build_samples(kern_prods, kern_sums, means, chol_covars)
        # output of log_cond_prob: (num_components, num_samples, batch_size, num_latent)
        # shape of loss_by_component: (num_components, batch_size, num_latent)
        loss_by_component = tf.reduce_mean(input_tensor=1.0 / (tf.exp(self.lik.log_cond_prob(
            train_outputs, latent_samples)) + 1e-7), axis=1)
        loss = tf.reduce_sum(input_tensor=weights[:, tf.newaxis, tf.newaxis] * loss_by_component, axis=0)
        return tf.reduce_sum(input_tensor=tfm.log(loss))

    def _build_ell(self, weights, means, chol_covars, inducing_inputs, kernel_chol, features,
                   train_outputs, _):
        """Construct the Expected Log Likelihood

        Args:
            weights: (num_components,)
            means: shape: (num_components, num_latents, num_inducing)
            chol_covars: shape: (num_components, num_latents, num_inducing[, num_inducing])
            inducing_inputs: (num_latents, num_inducing, input_dim)
            kernel_chol: (num_latents, num_inducing, num_inducing)
            train_inputs: (batch_size, input_dim)
            train_outputs: (batch_size, num_latents)
        Returns:
            Expected log likelihood as scalar
        """
        kern_prods, kern_sums = self._build_interim_vals(kernel_chol, inducing_inputs,
                                                         features['input'])
        # shape of `latent_samples`: (num_components, num_samples, batch_size, num_latents)
        latent_samples = self._build_samples(kern_prods, kern_sums, means, chol_covars)
        log_cond_prob = self.lik.log_cond_prob(train_outputs, latent_samples)
        ell_by_component = tf.reduce_sum(input_tensor=log_cond_prob, axis=[1, 2])

        # weighted sum of the components
        ell = util.mul_sum(weights, ell_by_component)
        return ell / self.args['num_samples']

    def _build_interim_vals(self, kernel_chol, inducing_inputs, train_inputs):
        """Helper function for `_build_ell`

        Args:
            kernel_chol: Tensor(num_latents, num_inducing, num_inducing)
            inducing_inputs: Tensor(num_latents, num_inducing, input_dim)
            train_inputs: Tensor(batch_size, input_dim)
        Returns:
            `kern_prods` (num_latents, batch_size, num_inducing)
            and `kern_sums` (num_latents, batch_size)
        """
        # shape of ind_train_kern: (num_latents, num_inducing, batch_size)

        kern_prods = [0.0 for _ in range(self.num_latents)]
        kern_sums = [0.0 for _ in range(self.num_latents)]

        for i in range(self.num_latents):
            ind_train_kern = self.cov[i](inducing_inputs[i, :, :], point2=train_inputs)
            # Compute A = Kxz.Kzz^(-1) = (Kzz^(-1).Kzx)^T.
            kern_prods[i] = tf.transpose(a=tf.linalg.cholesky_solve(kernel_chol[i, :, :], ind_train_kern))
            # We only need the diagonal components.
            kern_sums[i] = (self.cov[i].diag_cov_func(train_inputs) -
                            util.mul_sum(kern_prods[i], tf.linalg.transpose(ind_train_kern)))

        kern_prods = tf.stack(kern_prods, 0)
        kern_sums = tf.stack(kern_sums, 0)

        return kern_prods, kern_sums

    def _build_samples(self, kern_prods, kern_sums, means, chol_covars):
        """Produce samples according to the given distribution.

        Args:
            kern_prods: (num_latents, batch_size, num_inducing)
            kern_sums: (num_latents, batch_size)
            means: (num_components, num_latents, num_inducing)
            chol_covars: (num_components, num_latents, num_inducing[, num_inducing])
        Returns:
        """
        sample_means, sample_vars = self._build_sample_info(kern_prods, kern_sums, means,
                                                            chol_covars)
        batch_size = tf.shape(input=sample_means)[-2]
        norms = tf.random.normal([self.args['num_components'], self.args['num_samples'],
                                  batch_size, self.num_latents])
        return (sample_means[:, tf.newaxis, ...] + tf.sqrt(sample_vars)[:, tf.newaxis, ...] * norms)

    def _build_sample_info(self, kern_prods, kern_sums, means, chol_covars):
        """Get means and variances of a distribution

        Args:
            kern_prods: (num_latents, batch_size, num_inducing)
            kern_sums: (num_latents, batch_size)
            means: (num_components, num_latents, num_inducing)
            chol_covars: (num_components, num_latents, num_inducing[, num_inducing])
        Returns:
            sample_means (num_components, batch_size, num_latents),
            sample_vars (num_components, batch_size, num_latents)
        """
        if self.args['diag_post']:
            quad_form = util.mul_sum(kern_prods * chol_covars[..., tf.newaxis, :], kern_prods)
        else:
            full_covar = util.mat_square(chol_covars)  # same shape as chol_covars
            quad_form = util.mul_sum(util.matmul_br(kern_prods, full_covar), kern_prods)
        # shape: (num_components, num_latents, batch_size,1)
        sample_means = util.matmul_br(kern_prods, means[..., tf.newaxis])
        sample_vars = tf.linalg.transpose(kern_sums + quad_form)  # (num_components, x, num_latents)
        return tf.linalg.transpose(tf.squeeze(sample_means, -1)), sample_vars

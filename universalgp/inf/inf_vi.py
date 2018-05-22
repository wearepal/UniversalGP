"""
Variational inference for generic Gaussian process models
"""
import tensorflow as tf
from tensorflow.contrib.distributions import (MultivariateNormalDiag, MultivariateNormalTriL,
                                              matrix_diag_transform)
import numpy as np
from .. import util

JITTER = 1e-2

tf.app.flags.DEFINE_integer('num_components', 1,
                            'Number of mixture of Gaussians components')
tf.app.flags.DEFINE_integer('num_samples', 100,
                            'Number of samples for mean and variance estimate of likelihood')
tf.app.flags.DEFINE_boolean('diag_post', False,
                            'Whether the posterior is diagonal or not')
tf.app.flags.DEFINE_boolean('optimize_inducing', True,
                            'Whether to optimize the inducing inputs in training')
tf.app.flags.DEFINE_boolean('use_loo', False,
                            'Whether to use the LOO (leave one out) loss (for hyper parameters)')


class Variational:
    """
    Defines inference for Variational Inference
    """

    def __init__(self, cov_func, lik_func, num_train, inducing_inputs, args):
        """Create a new variational inference object which will keep track of all variables.

        Args:
            cov_func: covariance function (kernel function)
            lik_func: likelihood function
            num_train: the number of training examples
            inducing_inputs: the initial values for the inducing_inputs or just the number of
                             inducing inputs
            args: additional parameters: num_components, diag_post, use_loo, num_samples,
                  optimize_inducing
        """

        # self.mean = mean_func
        self.cov = cov_func
        self.lik = lik_func
        self.num_train = num_train
        self.num_latents = len(self.cov)
        self.args = args

        # Initialize inducing inputs if they are provided
        if isinstance(inducing_inputs, int):
            # Only the number of inducing inputs is given -> just specify the shape
            num_inducing = inducing_inputs
            inducing_params = {'shape': [self.num_latents, num_inducing, self.cov[0].input_dim],
                               'dtype': tf.float32}
        else:
            # Repeat the inducing inputs for all latent processes if we haven't been given
            # individually specified inputs per process.
            if inducing_inputs.ndim == 2:
                inducing_inputs = np.tile(inducing_inputs[np.newaxis, :, :],
                                          reps=[self.num_latents, 1, 1])
            # Initialize with the given values
            inducing_params = {'initializer': tf.constant(inducing_inputs, dtype=tf.float32)}
            num_inducing = inducing_inputs.shape[-2]

        num_components = args['num_components']
        # Initialize all variables
        with tf.variable_scope(None, "variational_inference"):
            # Define all parameters that get optimized directly in raw form. Some parameters get
            # transformed internally to maintain certain pre-conditions.

            self.inducing_inputs = tf.get_variable("inducing_inputs", **inducing_params)

            zeros = tf.zeros_initializer(dtype=tf.float32)
            self.raw_weights = tf.get_variable("raw_weights", [num_components], initializer=zeros)
            self.means = tf.get_variable("means", [num_components, self.num_latents, num_inducing],
                                         initializer=zeros)
            if args['diag_post']:
                self.raw_covars = tf.get_variable("raw_covars",
                                                  [num_components, self.num_latents, num_inducing],
                                                  initializer=tf.ones_initializer())
            else:
                self.raw_covars = tf.get_variable(
                    "raw_covars",
                    shape=[num_components, self.num_latents] + util.tri_vec_shape(num_inducing),
                    initializer=zeros)

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
            chol_covars = matrix_diag_transform(triangle, transform=tf.nn.softplus)

        # Build the matrices of covariances between inducing inputs.
        kernel_mat = tf.stack([self.cov[i].cov_func(self.inducing_inputs[i, :, :])
                               for i in range(self.num_latents)], 0)
        jitter = JITTER * tf.eye(tf.shape(self.inducing_inputs)[-2])

        kernel_chol = tf.cholesky(kernel_mat + jitter)
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
        batch_size = tf.to_float(tf.shape(outputs)[0])
        nelbo = -((batch_size / self.num_train) * (entropy + cross_ent) + ell)

        # Variables that will be changed during training
        vars_to_train = [self.means, self.raw_covars, self.raw_weights]
        if self.args['optimize_inducing']:
            vars_to_train += [self.inducing_inputs]

        obj_funcs = dict(elbo=-nelbo, entropy=(batch_size / self.num_train) * entropy,
                         cross_ent=(batch_size / self.num_train) * cross_ent, ell=ell)
        if self.args['use_loo']:
            # Compute LOO loss only when necessary
            loo_loss = self._build_loo_loss(weights, self.means, chol_covars, self.inducing_inputs,
                                            kernel_chol, features, outputs)
            return {**obj_funcs, 'NELBO': tf.squeeze(nelbo), 'LOO_VARIATIONAL': loo_loss,
                    'loss': tf.squeeze(nelbo) + loo_loss}, vars_to_train
        return {**obj_funcs, 'loss': tf.squeeze(nelbo)}, vars_to_train

    def predict(self, test_inputs):
        """Construct predictive distribution

        weights: (num_components,)
        means: shape: (num_components, num_latents, num_inducing)
        chol_covars: shape: (num_components, num_latents, num_inducing[, num_inducing])
        inducing_inputs: (num_latents, num_inducing, input_dim)
        kernel_chol: (num_latents, num_inducing, num_inducing)

        Args:
            test_inputs: (batch_size, input_dim)
        Returns:
            means and variances of the predictive distribution
        """
        # Transform all raw variables into their internal form.
        weights, chol_covars, kernel_chol = self._transform_variables()

        kern_prods, kern_sums = self._build_interim_vals(kernel_chol, self.inducing_inputs,
                                                         test_inputs['input'])
        sample_means, sample_vars = self._build_sample_info(kern_prods, kern_sums, self.means,
                                                            chol_covars)
        pred_means, pred_vars = self.lik.predict(sample_means, sample_vars)

        # Compute the mean and variance of the gaussian mixture from their components.
        # weights = tf.expand_dims(tf.expand_dims(weights, 1), 1)
        weights = weights[:, tf.newaxis, tf.newaxis]
        weighted_means = tf.reduce_sum(weights * pred_means, 0)
        weighted_vars = (tf.reduce_sum(weights * (pred_means ** 2 + pred_vars), 0) -
                         tf.reduce_sum(weights * pred_means, 0) ** 2)
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
            variational_dist = MultivariateNormalDiag(
                means, tf.sqrt(chol_covars[tf.newaxis, ...] + chol_covars[:, tf.newaxis, ...]))
        else:
            if self.args['num_components'] == 1:
                # Use the fact that chol(S + S) = sqrt(2) * chol(S)
                chol_covars_sum = tf.sqrt(2.) * chol_covars[tf.newaxis, ...]
            else:
                # Here we use the original component_covar directly
                # TODO: Can we just stay in cholesky space somehow?
                component_covar = util.mat_square(chol_covars)
                chol_covars_sum = tf.cholesky(component_covar[tf.newaxis, ...] +
                                              component_covar[:, tf.newaxis, ...])
            # The class MultivariateNormalTriL only accepts cholesky decompositions of covariances
            variational_dist = MultivariateNormalTriL(means[tf.newaxis, ...], chol_covars_sum)

        # compute log probability of all means in all normal distributions
        # then sum over all latent functions
        # shape of log_normal_probs: (num_components, num_components)
        log_normal_probs = tf.reduce_sum(variational_dist.log_prob(means[:, tf.newaxis, ...]), -1)

        # Now compute the entropy.
        # broadcast `weights` into dimension 1, then do `logsumexp` in that dimension
        weighted_logsumexp_probs = tf.reduce_logsumexp(tf.log(weights) + log_normal_probs, 1)
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
            trace = tf.trace(util.cholesky_solve_br(kernel_chol, tf.matrix_diag(chol_covars)))
        else:
            trace = tf.reduce_sum(util.mul_sum(util.cholesky_solve_br(kernel_chol, chol_covars),
                                               chol_covars), axis=-1)

        # sum_val has the same shape as weights
        gaussian = MultivariateNormalTriL(means, kernel_chol)
        sum_val = tf.reduce_sum(gaussian.log_prob([0.0]) - 0.5 * trace, -1)

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
        loss_by_component = tf.reduce_mean(1.0 / (tf.exp(self.lik.log_cond_prob(
            train_outputs, latent_samples)) + 1e-7), axis=1)
        loss = tf.reduce_sum(weights[:, tf.newaxis, tf.newaxis] * loss_by_component, axis=0)
        return tf.reduce_sum(tf.log(loss))

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
        ell_by_component = tf.reduce_sum(log_cond_prob, axis=[1, 2])

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
            ind_train_kern = self.cov[i].cov_func(inducing_inputs[i, :, :], train_inputs)
            # Compute A = Kxz.Kzz^(-1) = (Kzz^(-1).Kzx)^T.
            kern_prods[i] = tf.transpose(tf.cholesky_solve(kernel_chol[i, :, :], ind_train_kern))
            # We only need the diagonal components.
            kern_sums[i] = (self.cov[i].diag_cov_func(train_inputs) -
                            util.mul_sum(kern_prods[i], tf.matrix_transpose(ind_train_kern)))

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
        batch_size = tf.shape(sample_means)[-2]
        norms = tf.random_normal([self.args['num_components'], self.args['num_samples'],
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
        sample_vars = tf.matrix_transpose(kern_sums + quad_form)  # (num_components, x, num_latents)
        return tf.matrix_transpose(tf.squeeze(sample_means, -1)), sample_vars

    def get_all_variables(self):
        """Returns all variables, not just the ones that are trained."""
        return [self.means, self.raw_covars, self.raw_weights, self.inducing_inputs]

"""
Leave-one-out inference for Gaussian process models

Reference:
Carl Edward Rasmussen and Christopher K. I. Williams
The MIT Press, 2006. ISBN 0-262-18253-X. p116.
"""
import tensorflow as tf
from .. import util
from .base import Inference

JITTER = 1e-2


class Loo(Inference):
    """Class for inference based on LOO."""

    def build(self, input_shape):
        input_dim = int(input_shape[1])
        self.lik, self.cov = util.construct_lik_and_cov(self, self.args, self.lik_name, input_dim,
                                                        self.output_dim)
        self.sn = self.lik.sn
        self.train_inputs = self.add_variable('train_inputs', [self.num_train, input_dim],
                                              trainable=False)
        self.train_outputs = self.add_variable('train_outputs', [self.num_train, self.output_dim],
                                               trainable=False)
        super().build(input_shape)

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
        assignments = []
        if is_train:
            # During training, we have to store the training data to compute predictions later on
            assignments.append(self.train_inputs.assign(inputs))
            assignments.append(self.train_outputs.assign(outputs))

        with tf.control_dependencies(assignments):  # this ensures that the assigments are executed
            chol, alpha = self._build_interim_vals(inputs, outputs)
        # precision = inv(kxx)
        precision = tf.cholesky_solve(chol, tf.eye(tf.shape(inputs)[-2]))
        precision_diag = tf.matrix_diag_part(precision)

        loo_fmu = outputs - alpha / precision_diag   # GMPL book eq. 5.12
        loo_fs2 = 1.0 / precision_diag               # GMPL book eq. 5.12

        # log probability (lp), also called log pseudo-likelihood)
        lp = self._build_loo(outputs, loo_fmu, loo_fs2)

        return {'loss': -lp, 'LP': lp}

    def predict(self, test_inputs):
        """Build graph for computing predictive mean and variance

        Args:
            test_inputs: test inputs
        Returns:
            predictive mean and variance
        """
        return self.__call__(test_inputs['input'])

    def call(self, inputs, **_):
        chol, alpha = self._build_interim_vals(self.train_inputs, self.train_outputs)

        # kxx_star (num_latent, num_train, num_test)
        kxx_star = self.cov[0].cov_func(self.train_inputs, inputs)
        # f_star_mean (num_latent, num_test, 1)
        f_star_mean = tf.matmul(kxx_star, alpha, transpose_a=True)
        # Kx_star_x_star (num_latent, num_test)
        kx_star_x_star = self.cov[0].cov_func(inputs)
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

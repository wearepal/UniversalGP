import numpy as np
import tensorflow as tf

from .. import util


class Normal:
    """Class that represents a normal distribution."""

    def __init__(self, mean, covar):
        """
        Args:
            mean: the mean of the normal distribution
            covar: the covariance of the normal distribution
        """
        self.mean = mean
        self.covar = covar


class CholNormal(Normal):
    def prob(self, val):
        return tf.exp(self.log_prob(val))

    def log_prob(self, val):
        """Log probability

        `self.mean`: shape: (num_components, num_latent, num_inducing)
        `self.covar`: shape: ([num_components, ]num_latent, num_inducing, num_inducing)

        Args:
            val: scalar
        Returns:
            Tensor with shape (num_components, num_latent)
        """
        dim = tf.to_float(tf.shape(self.mean)[-1])
        diff = (val - self.mean)[..., tf.newaxis]  # shape: (num_components, num_latent, num_inducing, 1)
        quad_form = tf.reduce_sum(diff * util.cholesky_solve_br(self.covar, diff), axis=[-2, -1])
        return -0.5 * (dim * tf.log(2.0 * np.pi) + util.log_cholesky_det(self.covar) + quad_form)


class DiagNormal(Normal):
    def prob(self, val):
        return tf.exp(self.log_prob(val))

    def log_prob(self, val):
        """Log probability for `val`.

        `self.mean`: shape: (num_components, num_latent, num_inducing)
        `self.covar`: shape: (num_components, num_components, num_latent, num_inducing)

        Args:
            val: shape: (num_components, 1, num_latent, num_inducing)
        Returns:
            Tensor with shape (num_components, num_latent)
        """
        dim = tf.to_float(tf.shape(self.mean)[-1])
        # the following could be replaced by tensordot except tensordot does not broadcast
        quad_form = tf.reduce_sum((val - self.mean) ** 2 / self.covar, axis=-1)
        return -0.5 * (dim * tf.log(2.0 * np.pi) + tf.reduce_sum(tf.log(self.covar), -1) + quad_form)

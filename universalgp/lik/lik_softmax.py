import tensorflow as tf

from .base import Likelihood
from .. import util

tf.compat.v1.app.flags.DEFINE_integer(
    'num_samples_pred', 2000, 'Number of samples for mean and variance estimate for prediction')


class LikelihoodSoftmax(Likelihood):
    """Softmax likelihood used for multi-class classification"""
    def build(self, input_shape):
        self.num_samples = self.args['num_samples_pred']
        super().build(input_shape)

    @staticmethod
    def log_cond_prob(outputs, latent):
        # shape of `outputs`: (batch_size, output_dim)
        # shape of `latent`: (num_components, num_samples, batch_size, num_latent)
        # return tf.reduce_sum(outputs * latent, -1) - tf.reduce_logsumexp(latent, -1)
        # TODO(thomas): the batch_size and output_dim is usually not known
        outputs_tiled = util.broadcast(outputs, latent)
        return -tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(outputs_tiled), logits=latent)

    def call(self, latent_means, variances=None):
        """Given the distribution over the latent functions, what is the likelihood distribution?

        Args:
            latent_means: (num_components, batch_size, num_latent)
            variances: (num_components, batch_size, num_latent)
        Returns:
            `pred_means` and `pred_vars`
        """
        if variances is None:
            raise ValueError("variances should not be None")
        # Generate samples to estimate the expected value and variance of outputs.
        num_components = latent_means.shape[0]
        num_points = tf.shape(input=latent_means)[1]
        output_dims = tf.shape(input=latent_means)[2]
        latent = (latent_means[:, tf.newaxis, ...] + tf.sqrt(variances)[:, tf.newaxis, ...] *
                  tf.random.normal([num_components, self.num_samples, num_points, output_dims]))
        # Compute the softmax of all generated latent values in a stable fashion.
        # softmax = tf.exp(latent - tf.reduce_logsumexp(latent, 2, keep_dims=True))
        softmax = tf.nn.softmax(latent)

        # Estimate the expected value of the softmax and the variance through sampling.
        pred_means = tf.reduce_mean(input_tensor=softmax, axis=1)
        pred_vars = tf.reduce_sum(
            input_tensor=(softmax - pred_means[:, tf.newaxis, ...]) ** 2, axis=1) / (self.num_samples - 1.0)

        return pred_means, pred_vars

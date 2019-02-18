"""
Logistic likelihood function
"""

import tensorflow as tf

from .. import util


class LikelihoodLogistic:
    """Logistic likelihood function """
    def __init__(self, _, args):
        self.num_samples = args['num_samples_pred']

    @staticmethod
    def log_cond_prob(outputs, latent):
        # return latent * (outputs - 1) - tfm.log(1 + tf.exp(-latent))
        # in order to use the logistic likelihood,
        # the last dimension of both output and latent function should be 1
        latent = tf.squeeze(latent, axis=-1)
        outputs_expanded = util.broadcast(tf.squeeze(outputs, axis=-1), latent)
        return -tf.nn.sigmoid_cross_entropy_with_logits(labels=outputs_expanded, logits=latent)

    def predict(self, latent_means, latent_vars):
        """Given the distribution over the latent functions, what is the likelihood distribution?

        Args:
            latent_means: (num_components, batch_size, num_latent)
            latent_vars: (num_components, batch_size, num_latent)
        Returns:
            `pred_means` and `pred_vars`
        """
        # Generate samples to estimate the expected value and variance of outputs.
        num_components = latent_means.shape[0]
        num_points = tf.shape(input=latent_means)[1]
        latent = (latent_means[:, tf.newaxis, ...] + tf.sqrt(latent_vars)[:, tf.newaxis, ...] *
                  tf.random.normal([num_components, self.num_samples, num_points, 1]))
        # Compute the logistic function
        # logistic = 1.0 / (1.0 + tf.exp(-latent))
        logistic = tf.nn.sigmoid(latent)

        # Estimate the expected value of the softmax and the variance through sampling.
        pred_means = tf.reduce_mean(input_tensor=logistic, axis=1, keepdims=True)
        pred_vars = tf.reduce_sum(input_tensor=(logistic - pred_means) ** 2, axis=1) / (self.num_samples - 1.0)

        return tf.squeeze(pred_means, 1), pred_vars

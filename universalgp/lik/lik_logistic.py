import tensorflow as tf

from .. import util


class LikelihoodLogistic:
    def __init__(self, args):
        self.num_samples = args['num_samples_pred']

    def log_cond_prob(self, outputs, latent):
        # return latent * (outputs - 1) - tf.log(1 + tf.exp(-latent))
        # in order to use the logistic likelihood, the last dimension of both output and latent function should be 1
        latent = tf.squeeze(latent, axis=-1)
        outputs_expanded = util.broadcast(tf.squeeze(outputs, axis=-1), latent)
        return -tf.nn.sigmoid_cross_entropy_with_logits(labels=outputs_expanded, logits=latent)

    def get_params(self):
        return []

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
        num_points = tf.shape(latent_means)[1]
        latent = (latent_means[:, tf.newaxis, ...] + tf.sqrt(latent_vars)[:, tf.newaxis, ...] *
                  tf.random_normal([num_components, self.num_samples, num_points, 1]))
        # Compute the logistic function
        # logistic = 1.0 / (1.0 + tf.exp(-latent))
        logistic = tf.sigmoid(latent)

        # Estimate the expected value of the softmax and the variance through sampling.
        pred_means = tf.reduce_mean(logistic, 1)
        pred_vars = tf.reduce_sum((logistic - pred_means) ** 2, 1) / (self.num_samples - 1.0)

        return pred_means, pred_vars

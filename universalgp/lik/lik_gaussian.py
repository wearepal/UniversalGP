"""
Gaussian likelihood function
"""

import numpy as np
import tensorflow as tf
from tensorflow import math as tfm

tf.compat.v1.app.flags.DEFINE_float('sn', 1.0, 'Initial standard dev for the Gaussian likelihood')


class LikelihoodGaussian:
    """Gaussian likelihood function """
    def __init__(self, variables, args):
        init_sn = tf.keras.initializers.Constant(args['sn']) if 'sn' in args else None
        with tf.compat.v1.variable_scope(None, "gaussian_likelihood"):
            self.sn = variables.add_variable("std_dev", [], initializer=init_sn, dtype=tf.float32)

    def log_cond_prob(self, y, mu):
        var = self.sn ** 2
        log_cond_prob_per_latent_fs = -0.5 * tfm.log(2. * np.pi * var) - ((y - mu)**2) / (2. * var)
        return tf.reduce_sum(input_tensor=log_cond_prob_per_latent_fs, axis=-1)  # sum over latent functions

    def predict(self, means, variances):
        return means, variances + self.sn ** 2

    @staticmethod
    def pred_log_prob(y, pred_mean, pred_var):
        return -0.5 * tfm.log(2.0 * np.pi * pred_var) - ((y - pred_mean) ** 2) / (2.0 * pred_var)

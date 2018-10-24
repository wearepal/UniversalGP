"""
Gaussian likelihood function
"""

import numpy as np
import tensorflow as tf
from tensorflow import math as tfm

tf.app.flags.DEFINE_float('sn', 1.0, 'Initial standard dev for the Gaussian likelihood')


class LikelihoodGaussian:
    """Gaussian likelihood function """
    def __init__(self, gp_obj, args):
        init_sn = tf.constant_initializer(args['sn'], dtype=tf.float32) if 'sn' in args else None
        with tf.variable_scope(None, "gaussian_likelihood"):
            self.sn = gp_obj.add_variable("std_dev", [], initializer=init_sn)

    def log_cond_prob(self, y, mu):
        var = self.sn ** 2
        log_cond_prob_per_latent_fs = -0.5 * tfm.log(2. * np.pi * var) - ((y - mu)**2) / (2. * var)
        return tf.reduce_sum(log_cond_prob_per_latent_fs, -1)  # sum over latent functions

    def get_params(self):
        return [self.sn]

    def predict(self, means, variances):
        return means, variances + self.sn ** 2

    @staticmethod
    def pred_log_prob(y, pred_mean, pred_var):
        return -0.5 * tfm.log(2.0 * np.pi * pred_var) - ((y - pred_mean) ** 2) / (2.0 * pred_var)

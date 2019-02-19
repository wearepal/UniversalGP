"""
Gaussian likelihood function
"""

import numpy as np
import tensorflow as tf
from tensorflow import math as tfm

from .base import Likelihood

tf.compat.v1.app.flags.DEFINE_float('sn', 1.0, 'Initial standard dev for the Gaussian likelihood')


class LikelihoodGaussian(Likelihood):
    """Gaussian likelihood function """
    def build(self, input_shape):
        init_sn = tf.keras.initializers.Constant(self.args['sn']) if 'sn' in self.args else None
        self.sn = self.add_variable("std_dev", [], initializer=init_sn, dtype=tf.float32)
        super().build(input_shape)

    def log_cond_prob(self, y, mu):
        var = self.sn ** 2
        log_cond_prob_per_latent_fs = -0.5 * tfm.log(2. * np.pi * var) - ((y - mu)**2) / (2. * var)
        return tf.reduce_sum(input_tensor=log_cond_prob_per_latent_fs, axis=-1)  # sum over latent functions

    def call(self, means, variances=None):
        if variances is None:
            raise ValueError("variances should not be None")
        return means, variances + self.sn ** 2

    @staticmethod
    def pred_log_prob(y, pred_mean, pred_var):
        return -0.5 * tfm.log(2.0 * np.pi * pred_var) - ((y - pred_mean) ** 2) / (2.0 * pred_var)

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 18:56:08 2018

@author: zc223
"""

import numpy as np
import tensorflow as tf

tf.app.flags.DEFINE_float('sn', 1.0, 'Initial standard dev for the Gaussian likelihood')


class LikelihoodGaussian:
    def __init__(self, args):
        init_sn = tf.constant_initializer(args['sn'], dtype=tf.float32) if 'sn' in args else None
        self.sn = tf.get_variable("Likelihood_param", [], initializer=init_sn)

    def log_cond_prob(self, y, mu):
        var = self.sn ** 2
        log_cond_prob_per_latent_func = -0.5 * tf.log(2.0 * np.pi * var) - ((y - mu) ** 2) / (2.0 * var)
        return tf.reduce_sum(log_cond_prob_per_latent_func, -1)  # sum over latent functions

    def get_params(self):
        return [self.sn]

    def predict(self, means, variances):
        return means, variances + self.sn ** 2

    @staticmethod
    def pred_log_prob(y, pred_mean, pred_var):
        return -0.5 * tf.log(2.0 * np.pi * pred_var) - ((y - pred_mean) ** 2) / (2.0 * pred_var)

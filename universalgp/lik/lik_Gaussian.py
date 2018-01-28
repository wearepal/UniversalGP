# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 18:56:08 2018

@author: zc223
"""

import numpy as np
import tensorflow as tf


class LikelihoodGaussian:
    def __init__(self, sn=1.0):
        self.sn = tf.Variable(sn)

    def log_cond_prob(self, y, mu):
        var = self.sn ** 2
        return -0.5 * tf.log(2.0 * np.pi * var) - ((y - mu) ** 2) / (2.0 * var)

    def get_params(self):
        return [self.sn]

    def predict(self, means, variances):
        return means, variances + self.sn ** 2

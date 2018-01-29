#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 15:57:59 2018

@author: zc223
"""

import numpy as np
import tensorflow as tf
from .. import util


class SquaredExponential:

    def __init__(self, input_dim, length_scale=1.0, sf=1.0, iso_ard=False):

        # if iso_ard = False, consider ARD (automatic relevance determination) kernel,
        # else, consider ISO (isotropic ) kernel

        if iso_ard:
            self.length_scale = tf.Variable([length_scale], dtype=tf.float32)
        else:
            self.length_scale = tf.Variable(length_scale * tf.ones([input_dim]))

        self.sf = tf.Variable([sf], dtype=tf.float32)
        self.input_dim = input_dim

    def cov_func(self, point1, point2):
        kern = (self.sf ** 2) * tf.exp(-util.sq_dist(
            point1, point2) / 2.0 / (self.length_scale ** 2))
        return kern

    def diag_cov_func(self, point):
        return (self.sf ** 2) * tf.ones([tf.shape(point)[0]])

    def get_params(self):
        return [self.length_scale, self.sf]


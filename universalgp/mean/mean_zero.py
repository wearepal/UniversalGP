#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 18:56:08 2018

@author: zc223
"""

import tensorflow as tf


class ZeroOffset:
    def __init__(self):
        pass

    def mean_func(self, point=1):
        return tf.zeros(tf.shape(point))

    def get_params(self):
        return []

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 17:11:22 2018

@author: zc223
"""

import tensorflow as tf

MAX_DIST = 1e8


def sq_dist(point1, point2):
    square2 = tf.reduce_sum(point2 ** 2, 1)[:, tf.newaxis]
    square1 = tf.reduce_sum(point1 ** 2, 1)[:, tf.newaxis]
    distance = (square1 - 2 * point1 @ tf.transpose(point2) +
                tf.transpose(square2))
    distance = tf.clip_by_value(distance, 0.0, MAX_DIST)

    return distance


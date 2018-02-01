"""
Created on Tue Jan 23 17:11:22 2018

@author: zc223
"""

import tensorflow as tf
from . import util

MAX_DIST = 1e8


def sq_dist(point1, point2):
    """Compute the square distance between point1 and point2."""
    square2 = tf.reduce_sum(point2 ** 2, -1, keep_dims=True)
    square1 = tf.reduce_sum(point1 ** 2, -1, keep_dims=True)
    distance = (square1 - 2 * util.matmul_br(point1, point2, transpose_b=True) + tf.matrix_transpose(square2))

    # this ensures that exp(-distance) will never get too small
    distance = tf.clip_by_value(distance, 0.0, MAX_DIST)

    return distance

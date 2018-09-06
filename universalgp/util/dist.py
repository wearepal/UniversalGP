"""
Created on Tue Jan 23 17:11:22 2018

@author: zc223
"""

import tensorflow as tf
from . import util

MAX_DIST = 1e8
EPS = 1e-8


def dist(point1, point2):
    """Compute the distance between point1 and point2."""
    expanded1 = point1[..., tf.newaxis, :]
    expanded2 = point2[..., tf.newaxis, :, :]
    return expanded1 - expanded2


def sq_dist(point1, point2):
    """Compute the square distance between point1 and point2."""
    distance_vectors = dist(point1, point2)

    squared_distance = tf.reduce_sum(distance_vectors**2, axis=-1)

    # distance = tf.linalg.norm(distance_vectors, ord=2, axis=-1)
    # squared_distance = distance**2

    # this ensures that exp(-distance) will never get too small
    return tf.clip_by_value(squared_distance, 0.0, MAX_DIST)


def manhatten_dist(point1, point2):
    distance_vectors = dist(point1, point2)
    distance = tf.linalg.norm(distance_vectors, ord=1, axis=-1)
    # squared_distance = tf.reduce_sum(tf.abs(distance_vectors), axis=-1)

    # this ensures that exp(-distance) will never get too small
    return tf.clip_by_value(distance, 0.0, MAX_DIST)


def euclidean_dist(point1, point2):
    distance_vectors = dist(point1, point2)
    # distance = tf.linalg.norm(distance_vectors, ord=2, axis=-1)
    sq_distance = tf.reduce_sum(distance_vectors ** 2, axis=-1)
    distance = tf.sqrt(sq_distance + EPS)

    # this ensures that exp(-distance) will never get too small
    return tf.clip_by_value(distance, 0.0, MAX_DIST)

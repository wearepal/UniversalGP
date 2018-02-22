import numpy as np
import tensorflow as tf

from universalgp import util

import tensorflow.contrib.eager as tfe

try:
    tfe.enable_eager_execution()
except ValueError:
    pass


def broadcast(tensor_a, tensor_b):
    broadcasted = util.broadcast(tf.constant(tensor_a, dtype=tf.int32), tf.constant(tensor_b, dtype=tf.int32))
    return broadcasted.numpy()


def matmul_br(tensor_a, tensor_b, transpose_b=False):
    product = util.matmul_br(tf.constant(tensor_a, dtype=tf.int32), tf.constant(tensor_b, dtype=tf.int32),
                             transpose_b=transpose_b)
    return product.numpy()


class TestBroadcast:
    def test_broadcast(self):
        a = np.array([1, 2, 3, 4])
        b = np.array([
                     [[1, 1, 1, 1],
                      [1, 1, 1, 1]],

                     [[2, 2, 2, 2],
                      [2, 2, 2, 2]],

                     [[3, 3, 3, 3],
                      [3, 3, 3, 3]]
                     ])
        c = np.array([
                     [[1, 2, 3, 4],
                      [1, 2, 3, 4]],

                     [[1, 2, 3, 4],
                      [1, 2, 3, 4]],

                     [[1, 2, 3, 4],
                      [1, 2, 3, 4]]
                     ])
        np.testing.assert_equal(broadcast(a, b), c)

    def test_matmul_br(self):
        a = np.array([[1, 2, 3, 4]])
        b = np.array([
                     [[1, 1, 1, 1],
                      [1, 1, 1, 1]],

                     [[2, 2, 2, 2],
                      [2, 2, 2, 2]],

                     [[3, 3, 3, 3],
                      [3, 3, 3, 3]]
                     ])
        c = np.array([
                      [[10, 10]],

                      [[20, 20]],

                      [[30, 30]]
                     ])
        np.testing.assert_equal(matmul_br(a, b, transpose_b=True), c)
        np.testing.assert_equal(matmul_br(b, a, transpose_b=True), np.transpose(c, [0, 2, 1]))

    def test_matmul_br_2(self):
        a = np.reshape(np.array([1] * 12 + [-1] * 12), [2, 4, 3])
        b = np.reshape(np.array([2] * 6 + [3] * 6 + [4] * 6 + [5] * 6), [2, 2, 3, 2])
        c = np.array([[[[6, 6],
                        [6, 6],
                        [6, 6],
                        [6, 6]],

                       [[-9, -9],
                        [-9, -9],
                        [-9, -9],
                        [-9, -9]]],


                      [[[12, 12],
                        [12, 12],
                        [12, 12],
                        [12, 12]],

                       [[-15, -15],
                        [-15, -15],
                        [-15, -15],
                        [-15, -15]]]])
        np.testing.assert_equal(matmul_br(a, b), c)

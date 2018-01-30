# Copyright 2016 James Hensman, alexggmatthews
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
A collection of functions for tensorflow.
"""

import tensorflow as tf
import numpy as np


def tri_vec_shape(N):
    return [N * (N + 1) // 2]


def vec_to_tri(vectors):
    """
    Takes a DxM tensor vectors and maps it to a D x matrix_size x matrix_size
    tensor where the lower triangle of each matrix_size x matrix_size matrix
    is constructed by unpacking each M-vector.
    """
    M = vectors.shape[1].value
    N = int(np.floor(0.5 * np.sqrt(M * 8. + 1) - 0.5))
    assert N * (N + 1) == 2 * M  # check M is a valid triangle number.
    indices = tf.constant(np.stack(np.tril_indices(N)).T, dtype=tf.int32)

    def vec_to_tri_vector(vector):
        return tf.scatter_nd(indices=indices, shape=[N, N], updates=vector)

    return tf.map_fn(vec_to_tri_vector, vectors)


# @ops.RegisterGradient("VecToTri")
# def _vec_to_tri_grad(op, grad):
#     return [tri_to_vec(grad)]


# @ops.RegisterShape("VecToTri")
# def _vec_to_tri_shape(op):
#     in_shape = op.inputs[0].get_shape().with_rank(2)
#     M = in_shape[1].value
#     if M is None:
#         k = None
#     else:
#         k = int((M * 8 + 1) ** 0.5 / 2.0 - 0.5)
#     shape = tf.TensorShape([in_shape[0], k, k])
#     return [shape]

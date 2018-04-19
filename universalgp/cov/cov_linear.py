"""
Linear kernel
"""

import tensorflow as tf
from .. import util

tf.app.flags.DEFINE_float('lin_kern_offset', 0.0, 'The offset of the linear kernel')
tf.app.flags.DEFINE_float('lin_kern_sb', 0.0, 'Uncertain offset of the model for linear kernel')
tf.app.flags.DEFINE_float('lin_kern_sv', 1.0, 'Variance of linear kernel')


class Linear:
    """Linear kernel"""
    def __init__(self, input_dim, args, name=None):
        """
        Args:
            input_dim: the number of input dimensions
        """
        self.input_dim = input_dim
        init_offset = tf.constant_initializer(args['lin_kern_offset'], dtype=tf.float32) if (
            'lin_kern_offset' in args) else None
        init_sb = tf.constant_initializer(args['lin_kern_sb'], dtype=tf.float32) if 'lin_kern_sb' in args else None
        init_sv = tf.constant_initializer(args['lin_kern_sv'], dtype=tf.float32) if 'lin_kern_sv' in args else None
        with tf.variable_scope(name, "cov_lin_parameters"):
            self.offset = tf.get_variable("offset", [input_dim], initializer=init_offset)
            self.sigma_b = tf.get_variable("sb", shape=[], initializer=init_sb)
            self.sigma_v = tf.get_variable("sv", shape=[], initializer=init_sv)

    def cov_func(self, point1, point2=None):
        """
        Args:
            point1: Tensor(input_dim) or Tensor(batch_size, input_dim)
            point2: Tensor(batch_size, input_dim)
        Returns:
            Tensor of shape (batch_size, batch_size)
        """
        offset_br = tf.reshape(self.offset, [1, self.input_dim])
        if point2 is None:
            point2 = point1

        return self.sigma_b**2 + self.sigma_v**2 * util.matmul_br(
            point1 - offset_br, point2 - offset_br, transpose_b=True)

    def diag_cov_func(self, points):
        """
        Args:
            points: Tensor(batch_size, input_dim)
        Returns:
            Tensor of shape (batch_size)
        """
        offset_br = tf.reshape(self.offset, [1, self.input_dim])
        return self.sigma_b**2 + self.sigma_v**2 * tf.reduce_sum((points - offset_br)**2)

    def get_params(self):
        return [self.offset, self.sigma_b, self.sigma_v]

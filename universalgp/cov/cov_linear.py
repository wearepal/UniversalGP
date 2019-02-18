"""
Linear kernel
"""
import tensorflow as tf

from .. import util
from .base import Covariance

tf.compat.v1.app.flags.DEFINE_float('lin_kern_offset', 0.0, 'The offset of the linear kernel')
tf.compat.v1.app.flags.DEFINE_float('lin_kern_sb', 0.0,
                                    'Uncertain offset of the model for linear kernel')
tf.compat.v1.app.flags.DEFINE_float('lin_kern_sv', 1.0, 'Variance of linear kernel')


class Linear(Covariance):
    """Linear kernel"""
    def build(self, input_shape):
        """
        Args:
            variables: object that stores the variables
            input_dim: the number of input dimensions
            args: dictionary with parameters
        """
        self.input_dim = int(input_shape[-1])
        if 'lin_kern_offset' in self.args:
            init_offset = tf.keras.initializers.Constant(
                self.args['lin_kern_offset'])
        else:
            init_offset = None

        init_sb = tf.keras.initializers.Constant(self.args['lin_kern_sb']) if (
            'lin_kern_sb' in self.args) else None
        init_sv = tf.keras.initializers.Constant(self.args['lin_kern_sv']) if (
            'lin_kern_sv' in self.args) else None
        self.offset = self.add_variable("offset", [self.input_dim], initializer=init_offset,
                                        dtype=tf.float32)
        self.sigma_b = self.add_variable("sb", shape=[], initializer=init_sb, dtype=tf.float32)
        self.sigma_v = self.add_variable("sv", shape=[], initializer=init_sv, dtype=tf.float32)
        super().build(input_shape)

    def call(self, point1, point2=None):
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

"""
Squared exponential kernel
"""
import tensorflow as tf

from .. import util
from .base import Covariance

tf.compat.v1.app.flags.DEFINE_float('length_scale', 1.0, 'Initial length scale for the kernel')
tf.compat.v1.app.flags.DEFINE_float('sf', 1.0, 'Initial standard dev for the kernel')
tf.compat.v1.app.flags.DEFINE_boolean(
    'iso', False, 'True to use an isotropic kernel otherwise use automatic relevance det')


class SquaredExponential(Covariance):
    """Squared exponential kernel"""
    def build(self, input_shape):
        self.input_dim = int(input_shape[-1])
        self.iso = self.args['iso']
        length = tf.keras.initializers.Constant(self.args['length_scale']) if (
            'length_scale' in self.args) else None
        sigma_f = tf.keras.initializers.Constant(self.args['sf']) if (
            'sf' in self.args) else None
        if not self.args['iso']:
            self.length_scale = self.add_variable("length_scale", [self.input_dim],
                                                  initializer=length, dtype=tf.float32)
        else:
            self.length_scale = self.add_variable("length_scale", shape=[], initializer=length,
                                                  dtype=tf.float32)
        self.sf = self.add_variable("sf", shape=[], initializer=sigma_f, dtype=tf.float32)
        super().build(input_shape)

    def call(self, point1, point2=None):
        """
        Args:
            point1: Tensor(input_dim) or Tensor(batch_size, input_dim)
            point2: Tensor(batch_size, input_dim)
        Returns:
            Tensor of shape (batch_size, batch_size)
        """
        length_scale_br = tf.reshape(self.length_scale, [1, 1 if self.iso else self.input_dim])
        if point2 is None:
            point2 = point1

        kern = self.sf**2 * tf.exp(-util.sq_dist(point1 / length_scale_br,
                                                 point2 / length_scale_br) / 2.0)
        return kern

    def diag_cov_func(self, points):
        """
        Args:
            points: Tensor(batch_size, input_dim)
        Returns:
            Tensor of shape (batch_size)
        """
        return self.sf ** 2 * tf.ones([tf.shape(input=points)[-2]])

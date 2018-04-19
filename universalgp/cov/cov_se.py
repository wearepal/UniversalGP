"""
Squared exponential kernel
"""
import tensorflow as tf
from .. import util

tf.app.flags.DEFINE_float('length_scale', 1.0, 'Initial length scale for the kernel')
tf.app.flags.DEFINE_float('sf', 1.0, 'Initial standard dev for the kernel')
tf.app.flags.DEFINE_boolean('iso', False, 'Whether to use an isotropic kernel otherwise use automatic relevance det')


class SquaredExponential:
    """Squared exponential kernel"""
    def __init__(self, input_dim, args, name=None):
        """
        Args:
            input_dim: the number of input dimensions
        """
        self.input_dim = input_dim
        self.iso = args['iso']
        init_len = tf.constant_initializer(args['length_scale'], dtype=tf.float32) if 'length_scale' in args else None
        init_sf = tf.constant_initializer(args['sf'], dtype=tf.float32) if 'sf' in args else None
        with tf.variable_scope(name, "cov_se_parameters"):
            if not args['iso']:
                self.length_scale = tf.get_variable("length_scale", [input_dim], initializer=init_len)
            else:
                self.length_scale = tf.get_variable("length_scale", shape=[], initializer=init_len)
            self.sf = tf.get_variable("sf", shape=[], initializer=init_sf)

    def cov_func(self, point1, point2=None):
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

        kern = self.sf ** 2 * tf.exp(-util.sq_dist(point1 / length_scale_br, point2 / length_scale_br) / 2.0)
        return kern

    def diag_cov_func(self, points):
        """
        Args:
            points: Tensor(batch_size, input_dim)
        Returns:
            Tensor of shape (batch_size)
        """
        return self.sf ** 2 * tf.ones([tf.shape(points)[-2]])

    def get_params(self):
        return [self.length_scale, self.sf]

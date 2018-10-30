"""
Squared exponential kernel
"""
import tensorflow as tf
from tensorflow import manip as tft
from .. import util

tf.app.flags.DEFINE_float('length_scale', 1.0, 'Initial length scale for the kernel')
tf.app.flags.DEFINE_float('sf', 1.0, 'Initial standard dev for the kernel')
tf.app.flags.DEFINE_boolean('iso', False,
                            'True to use an isotropic kernel otherwise use automatic relevance det')


class SquaredExponential:
    """Squared exponential kernel"""
    def __init__(self, variables, input_dim, args, name=None):
        """
        Args:
            variables: object that stores the variables
            input_dim: the number of input dimensions
            args: dictionary with parameters
        """
        self.input_dim = input_dim
        self.iso = args['iso']
        length = tf.constant_initializer(args['length_scale'], dtype=tf.float32) if (
            'length_scale' in args) else None
        sigma_f = tf.constant_initializer(args['sf'], dtype=tf.float32) if 'sf' in args else None
        with tf.variable_scope(name, "cov_se_parameters"):
            if not args['iso']:
                self.length_scale = variables.add_variable("length_scale", [input_dim],
                                                           initializer=length)
            else:
                self.length_scale = variables.add_variable("length_scale", shape=[],
                                                           initializer=length)
            self.sf = variables.add_variable("sf", shape=[], initializer=sigma_f)

    def cov_func(self, point1, point2=None):
        """
        Args:
            point1: Tensor(input_dim) or Tensor(batch_size, input_dim)
            point2: Tensor(batch_size, input_dim)
        Returns:
            Tensor of shape (batch_size, batch_size)
        """
        length_scale_br = tft.reshape(self.length_scale, [1, 1 if self.iso else self.input_dim])
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
        return self.sf ** 2 * tf.ones([tf.shape(points)[-2]])

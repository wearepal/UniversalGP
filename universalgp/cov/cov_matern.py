"""Matern Kernel

Matern covariance function with nu = d/2, where d = 1, 3, 5. For d=1 the function is also known as
the exponential covariance function or the Ornstein-Uhlenbeck covariance  in 1d. The covariance
function is:   k(x^p,x^q) = sf^2 * f( sqrt(d)*r ) * exp(-sqrt(d)*r)
with f(t)=1 for d=1, f(t)=1+t for d=3 and f(t)=1+t+t²/3 for d=5. Here r is the distance.
If this is a Automatic Relevance Determination (ARD) distance measure, then
r = sqrt((x^p-x^q)'*inv(P)*(x^p-x^q)), where the P matrix is diagonal with ARD parameters
ell_1^2,...,ell_D^2 where D is the dimension of the input space and sf2 is the signal variance.
The hyperparameters are:

hyp = [ log(ell_1)
        log(ell_2)
         ..
        log(ell_D)
        log(sf) ]
If this is not a ARD kernel, the hyperparameters are:

hyp = [ log(ell)
        log(sf) ]
"""
import tensorflow as tf
from .. import util

tf.app.flags.DEFINE_integer('order', 3, 'The order of matern function')
# tf.app.flags.DEFINE_float('length_scale', 1.0, 'Initial length scale for the kernel')
# tf.app.flags.DEFINE_float('sf', 1.0, 'Initial standard dev for the kernel')
# tf.app.flags.DEFINE_boolean('iso', False,
#                             'True to use an isotropic kernel else use automatic relevance det')


class Matern:
    """Matern covariance function"""
    def __init__(self, variables, input_dim, args, name=None):
        """
        Args:
            variables: object that stores the variables
            input_dim: the number of input dimensions
            args: dictionary with parameters
        """
        self.input_dim = input_dim
        self.iso = args['iso']
        self.order = args['order']
        init_len = tf.constant_initializer(args['length_scale'], dtype=tf.float32) if (
            'length_scale' in args) else None
        init_sf = tf.constant_initializer(args['sf'], dtype=tf.float32) if 'sf' in args else None
        with tf.variable_scope(name, "cov_matern_parameters"):
            if not args['iso']:
                self.length_scale = variables.add_variable("length_scale", [input_dim],
                                                           initializer=init_len)
            else:
                self.length_scale = variables.add_variable("length_scale", shape=[],
                                                           initializer=init_len)
            self.sf = variables.add_variable("sf", shape=[], initializer=init_sf)

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

        distance = util.euclidean_dist(point1 / length_scale_br, point2 / length_scale_br)
        latent = tf.sqrt(float(self.order)) * distance
        kern = self.sf ** 2 * tf.exp(- latent) * self._interim_f(latent)
        return kern

    def _interim_f(self, r):
        if self.order == 1:
            return 1.0

        if self.order == 3:
            return 1.0 + r

        if self.order == 5:
            return 1.0 + r * (1.0 + r / 3.0)
        raise ValueError(f"Unsupported order: {self.order}")

    def diag_cov_func(self, points):
        """
        Args:
            points: Tensor(batch_size, input_dim)
        Returns:
            Tensor of shape (batch_size)
        """
        return self.sf ** 2 * tf.ones([tf.shape(points)[-2]])

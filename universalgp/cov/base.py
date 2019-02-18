"""Base class of covariance functions"""
import tensorflow as tf

class Covariance(tf.keras.layers.Layer):
    """Base class of covariance functions"""
    def __init__(self, args, **kwargs):
        """
        Args:
            args: dictionary with parameters
        """
        self.args = args
        super().__init__(**kwargs)

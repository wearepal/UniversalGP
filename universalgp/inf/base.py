"""Base class for inference methods"""

import tensorflow as tf


class Inference(tf.keras.layers.Layer):
    """Base class for inference methods"""
    def __init__(self, args, lik_name, output_dim, num_train, **kwargs):
        self.args = args
        self.lik_name = lik_name
        self.output_dim = output_dim
        self.num_train = num_train
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super().get_config()
        base_config['args'] = self.args
        base_config['lik_name'] = self.lik_name
        base_config['output_dim'] = self.output_dim
        base_config['num_train'] = self.num_train
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

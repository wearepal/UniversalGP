"""Base class for inference methods"""

import tensorflow as tf


class Inference(tf.keras.layers.Layer):
    """Base class for inference methods"""
    def __init__(self, args, lik_name, output_dim, num_train, inducing_inputs, **kwargs):
        self.args = args
        self.lik_name = lik_name
        self.output_dim = output_dim
        self.num_train = num_train
        if isinstance(inducing_inputs, int):
            self.num_inducing = inducing_inputs
        else:
            self.inducing_inputs_init = inducing_inputs
            self.num_inducing = inducing_inputs.shape[-2]
        super().__init__(**kwargs)

    def inference(self, features, outputs, is_train):
        """Compute loss"""
        raise NotImplementedError("Implement `inference`")

    def prediction(self, test_inputs):
        """Return prediction for given inputs"""
        raise NotImplementedError("Implement `prediction`")

    def call(self, inputs, **_):
        return self._apply(inputs)

    def _apply(self, inputs):
        """Actual code that computes the predictions. Can be called by calling `apply`.

        This function is called by `call`. `call` is called by `__call__` which can be called by
        calling `apply`. Use `apply` to call this function while making sure that `build`
        is called along the way.

        It unfortunately has to be this complicated because `__call__` has to be called somewhere.
        """
        raise NotImplementedError("Implement `_apply`")

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
        base_config['inducing_inputs'] = self.num_inducing
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

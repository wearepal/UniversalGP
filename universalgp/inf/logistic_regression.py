"""Defines a logistic regression model to serve as a baseline"""

import tensorflow as tf
from tensorflow import manip as tft
from tensorflow import math as tfm

from .inf_vi_ybar import (VariationalWithS, debiasing_params_target_rate,
                          debiasing_params_target_tpr)


class LogReg(VariationalWithS):
    """Simple logistic regression model"""
    def __init__(self, cov_func, lik_func, num_train, inducing_inputs, args):
        self.args = args
        self.s_as_input = args['s_as_input']
        input_dim = cov_func[0].input_dim
        regularize_factor = 0.1
        # create the logistic regression model
        # this is just a single layer neural network. we use no activation function here,
        # but we use `sigmoid_cross_entropy_with_logits` for the loss function which means
        # there is implicitly the logistic function as the activation function.
        self._model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=(input_dim,), activation=None, use_bias=True,
                                  kernel_regularizer=tf.keras.regularizers.l2(regularize_factor),
                                  bias_regularizer=tf.keras.regularizers.l2(regularize_factor))
        ])

    def _logits(self, features):
        """Compute logits"""
        if self.s_as_input:
            inputs = tf.concat((features['input'], features['sensitive']), axis=1)
        else:
            inputs = features['input']
        return self._model(inputs)

    def inference(self, features, outputs, _):
        """Standard logistic regression loss"""
        logits = self._logits(features)
        # this loss function implicitly uses the logistic function on the output of the one layer
        log_cond_prob = -tf.nn.sigmoid_cross_entropy_with_logits(labels=outputs, logits=logits)
        l2_loss = self._l2_loss()
        regr_loss = -tf.reduce_mean(tf.squeeze(log_cond_prob), axis=-1)  # regression loss
        return ({'loss': regr_loss + l2_loss, 'regr_loss': regr_loss, 'l2_loss': l2_loss},
                self._model.trainable_variables)

    def predict(self, test_inputs):
        pred = tf.nn.sigmoid(self._logits(test_inputs))
        return pred, tf.zeros_like(pred)

    def _trainable_variables(self):
        return self._model.trainable_variables

    def get_all_variables(self):
        return self._model.variables

    def _l2_loss(self):
        return tf.add_n(self._model.losses)  # L2 regularization loss


class FairLogReg(LogReg):
    """Fair logistic regression for demographic parity"""
    def inference(self, features, outputs, is_train):
        """Inference for targeting ybar"""
        if not is_train:
            return super().inference(features, outputs, is_train)
        sens_attr = tf.cast(tf.squeeze(features['sensitive'], -1), dtype=tf.int32)
        out_int = tf.cast(tf.squeeze(outputs, -1), dtype=tf.int32)
        # likelihood for y=1
        lik1 = tf.squeeze(tf.nn.sigmoid(self._logits(features)), axis=-1)
        # likelihood for y=0
        lik0 = 1 - lik1
        lik = tf.stack((lik0, lik1), axis=-1)
        debias = self._debiasing_parameters()
        # `debias` has the shape (y, s, y'). we stack output and sensitive to (batch_size, 2)
        # then we use the last 2 values of that as indices for `debias`
        # shape of debias_per_example: (batch_size, output_dim, 2)
        debias_per_example = tft.gather_nd(debias, tf.stack((out_int, sens_attr), axis=-1))
        weighted_lik = debias_per_example * lik
        log_cond_prob = tfm.log(tf.reduce_sum(weighted_lik, axis=-1))
        regr_loss = -tf.reduce_mean(log_cond_prob)
        l2_loss = self._l2_loss()
        return ({'loss': regr_loss + l2_loss, 'regr_loss': regr_loss, 'l2_loss': l2_loss},
                self._trainable_variables())

    def _debiasing_parameters(self):
        return debiasing_params_target_rate(self.args)


class EqOddsLogReg(FairLogReg):
    """Fair logistic regression for equalized odds (or equality of opportunity)"""
    def _debiasing_parameters(self):
        return debiasing_params_target_tpr(self.args)

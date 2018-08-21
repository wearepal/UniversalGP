"""Defines a logistic regression model to serve as a baseline"""

import tensorflow as tf

from .inf_vi_ybar import VariationalYbar


class LogisticRegressionModel:
    """Generic functionality for logistic regression models"""
    def __init__(self, s_as_input, input_dim):
        self.s_as_input = s_as_input
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=(input_dim,), activation=None, use_bias=True)
        ])

    def logits(self, features):
        """Compute logits"""
        if self.s_as_input:
            inputs = tf.concat((features['input'], features['sensitive']), axis=1)
        else:
            inputs = features['input']
        return self.model(inputs)

    def predict(self, test_inputs):
        return tf.nn.sigmoid(self.logits(test_inputs))

    def get_all_variables(self):
        return self.model.variables


class LogisticRegression(VariationalYbar):
    """Fair logistic regression"""
    def __init__(self, cov_func, lik_func, num_train, inducing_inputs, args):
        input_dim = cov_func[0].input_dim
        self.args = args
        self.lr = LogisticRegressionModel(args['s_as_input'], input_dim)

    def inference(self, features, outputs, is_train):
        if is_train:
            sens_attr = tf.cast(tf.squeeze(features['sensitive'], -1), dtype=tf.int32)
            out_int = tf.cast(tf.squeeze(outputs, -1), dtype=tf.int32)
            # likelihood for y=1
            lik1 = tf.squeeze(self.lr.predict(features), axis=-1)
            # likelihood for y=0
            lik0 = 1 - lik1
            lik = tf.stack((lik0, lik1), axis=-1)
            debias = self._debiasing_parameters()
            # `debias` has the shape (y, s, y'). we stack output and sensitive to (batch_size, 2)
            # then we use the last 2 values of that as indices for `debias`
            # shape of debias_per_example: (batch_size, output_dim, 2)
            debias_per_example = tf.gather_nd(debias, tf.stack((out_int, sens_attr), axis=-1))
            weighted_lik = debias_per_example * lik
            log_cond_prob = tf.log(tf.reduce_sum(weighted_lik, axis=-1))
        else:
            logits = self.lr.logits(features)
            log_cond_prob = -tf.nn.sigmoid_cross_entropy_with_logits(labels=outputs,
                                                                     logits=logits)
            log_cond_prob = tf.squeeze(log_cond_prob, axis=-1)
        loss = -tf.reduce_mean(log_cond_prob)
        return {'loss': loss}, self.lr.model.trainable_variables

    def predict(self, test_inputs):
        pred = self.lr.predict(test_inputs)
        return pred, pred

    def get_all_variables(self):
        return self.lr.get_all_variables()

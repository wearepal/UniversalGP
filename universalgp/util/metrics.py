"""Defines methods for metrics"""

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
# TODO: split into several files
# TODO: after splitting it into several files in a separate dictionary it should be possible to get rid of MAPPING


def init_metrics(metric_flag, is_eager):
    """Initialize metrics

    Args:
        metric_flag: a string that contains the names of the metrics separated with commas
        is_eager: True if in eager execution
    Returns:
        a dictionary with the initialized metrics
    """
    metrics = {}
    if metric_flag == "":
        return metrics

    # TODO: allow metrics to be defined as a list instead of one long string
    for name in metric_flag.split(','):
        try:
            metric = MAPPING[name]
        except KeyError:
            raise ValueError(f"Unknown metric \"{name}\"")
        metrics[name] = metric(is_eager)
    return metrics


def update_metrics(metrics, features, labels, pred_mean):
    """Update metrics

    Args:
        metrics: a dictionary with the initialized metrics
        features: the input
        labels: the correct labels
        pred_mean: the predicted mean
        is_eager: True if in eager execution
    Returns:
        dictionary of update ops if `is_eager` is False
    """
    update_ops = {}
    for name, metric in metrics.items():
        update_op = metric.update(features, labels, pred_mean)
        if update_op is not None:
            update_ops[name] = update_op
    return update_ops


def record_metrics(metrics):
    """Print the result or record it in the summary

    Args:
        metrics: a dictionary with the updated metrics
    """
    for metric in metrics.values():
        metric.record()


class Metric:
    """Base class for metrics"""
    def __init__(self, is_eager):
        self.is_eager = is_eager

    def update(self, features, labels, pred_mean):
        """Update the metric based on the given input, label and prediction

        Args:
            features: the input
            labels: the correct labels
            pred_mean: the predicted mean
        Returns:
            update op if `is_eager` is False
        """
        pass

    def record(self):
        """Print the result or record it in the summary"""
        pass

    def _return_and_store(self, metric_op):
        """In graph mode stores the result in `self.result` and returns the op so that it can be updated

        Does currently nothing in eager mode.
        """
        if not self.is_eager:
            self.result = metric_op[0]
            return metric_op


class Rmse(Metric):
    """Root mean squared error"""
    def __init__(self, is_eager):
        super().__init__(is_eager)
        self.metric = tfe.metrics.Mean() if is_eager else tf.metrics.root_mean_squared_error

    def update(self, features, labels, pred_mean):
        if self.is_eager:
            self.metric((pred_mean - labels)**2)
        else:
            return self._return_and_store(self.metric(labels, pred_mean))

    def record(self):
        if self.is_eager:
            print(f"RMSE: {np.sqrt(self.metric.result())}")
        else:
            tf.summary.scalar('RMSE', self.result)


class Mae(Metric):
    """Mean absolute error"""
    display_name = "MAE"

    def __init__(self, is_eager):
        super().__init__(is_eager)
        self.mean = tfe.metrics.Mean() if is_eager else tf.metrics.mean

    def update(self, features, labels, pred_mean):
        return self._return_and_store(self.mean(tf.abs(pred_mean - labels)))

    def record(self):
        if self.is_eager:
            print(f"{self.display_name}: {self.mean.result()}")
        else:
            tf.summary.scalar(self.display_name, self.result)


class SoftAccuracy(Metric):
    """Accuracy for softmax output"""
    display_name = "Accuracy"

    def __init__(self, is_eager):
        super().__init__(is_eager)
        self.accuracy = tfe.metrics.Accuracy() if is_eager else tf.metrics.accuracy

    def update(self, features, labels, pred_mean):
        return self._return_and_store(self.accuracy(tf.argmax(labels, axis=1), tf.argmax(pred_mean, axis=1)))

    def record(self):
        if self.is_eager:
            print(f"{self.display_name}: {self.accuracy.result()}")
        else:
            tf.summary.scalar(self.display_name, self.result)


class LogisticAccuracy(SoftAccuracy):
    """Accuracy for output from the logistic function"""
    display_name = "Accuracy"

    def update(self, features, labels, pred_mean):
        return self._return_and_store(self.accuracy(tf.cast(labels, tf.int32), tf.cast(pred_mean > 0.5, tf.int32)))


class LogisticAccuracyYbar(SoftAccuracy):
    """Accuracy for output from the logistic function"""
    display_name = "Accuracy"

    def update(self, features, labels, pred_mean):
        return self._return_and_store(self.accuracy(features['ybar'], tf.cast(pred_mean > 0.5, tf.float32)))


class PredictionRateY1S0(Mae):
    """Acceptance Rate, group 1"""
    display_name = "Prediction_rate_y1_s0"

    def update(self, features, labels, pred_mean):
        accepted = tf.gather_nd(tf.cast(pred_mean > 0.5, tf.float32), tf.where(tf.equal(features['sensitive'], 0)))
        return self._return_and_store(self.mean(accepted))


class PredictionRateY1S1(Mae):
    """Acceptance Rate, group 2"""
    display_name = "Prediction_rate_y1_s1"

    def update(self, features, labels, pred_mean):
        accepted = tf.gather_nd(tf.cast(pred_mean > 0.5, tf.float32), tf.where(tf.equal(features['sensitive'], 1)))
        return self._return_and_store(self.mean(accepted))


class BaseRateY1S0(Mae):
    """Base acceptance rate, group 1"""
    display_name = "Base_rate_y1_s0"

    def update(self, features, labels, pred_mean):
        accepted = tf.gather_nd(labels, tf.where(tf.equal(features['sensitive'], 0)))
        return self._return_and_store(self.mean(accepted))


class BaseRateY1S1(Mae):
    """Base acceptance rate, group 2"""
    display_name = "Base_rate_y1_s1"

    def update(self, features, labels, pred_mean):
        accepted = tf.gather_nd(labels, tf.where(tf.equal(features['sensitive'], 1)))
        return self._return_and_store(self.mean(accepted))


class PredictionOddsYYbar1S0(Mae):
    """Opportunity P(yhat=1|s,ybar=1), group 1"""
    display_name = "Opportunity_s0"

    def update(self, features, labels, pred_mean):
        test_for_ybar1_s0 = tf.logical_and(tf.equal(features['ybar'], 1), tf.equal(features['sensitive'], 0))
        accepted = tf.gather_nd(tf.cast(pred_mean > 0.5, tf.float32), tf.where(test_for_ybar1_s0))
        return self._return_and_store(self.mean(accepted))


class PredictionOddsYYbar1S1(Mae):
    """Opportunity P(yhat=1|s,ybar=1), group 2"""
    display_name = "Opportunity_s1"

    def update(self, features, labels, pred_mean):
        test_for_ybar1_s1 = tf.logical_and(tf.equal(features['ybar'], 1), tf.equal(features['sensitive'], 1))
        accepted = tf.gather_nd(tf.cast(pred_mean > 0.5, tf.float32), tf.where(test_for_ybar1_s1))
        return self._return_and_store(self.mean(accepted))


class BaseOddsYYbar1S0(Mae):
    """Opportunity P(y=1|s,ybar=1), group 1"""
    display_name = "Base_opportunity_s0"

    def update(self, features, labels, pred_mean):
        test_for_ybar1_s0 = tf.logical_and(tf.equal(features['ybar'], 1), tf.equal(features['sensitive'], 0))
        accepted = tf.gather_nd(labels, tf.where(test_for_ybar1_s0))
        return self._return_and_store(self.mean(accepted))


class BaseOddsYYbar1S1(Mae):
    """Opportunity P(y=1|s,ybar=1), group 2"""
    display_name = "Base_opportunity_s1"

    def update(self, features, labels, pred_mean):
        test_for_ybar1_s1 = tf.logical_and(tf.equal(features['ybar'], 1), tf.equal(features['sensitive'], 1))
        accepted = tf.gather_nd(labels, tf.where(test_for_ybar1_s1))
        return self._return_and_store(self.mean(accepted))


class PredictionOddsYYbar0S0(Mae):
    """Opportunity P(yhat=1|s,ybar=1), group 1"""
    display_name = "Opportunity_s0"

    def update(self, features, labels, pred_mean):
        test_for_ybar0_s0 = tf.logical_and(tf.equal(features['ybar'], 0), tf.equal(features['sensitive'], 0))
        accepted = tf.gather_nd(tf.cast(pred_mean < 0.5, tf.float32), tf.where(test_for_ybar0_s0))
        return self._return_and_store(self.mean(accepted))


class PredictionOddsYYbar0S1(Mae):
    """Opportunity P(yhat=1|s,ybar=1), group 2"""
    display_name = "Opportunity_s1"

    def update(self, features, labels, pred_mean):
        test_for_ybar0_s1 = tf.logical_and(tf.equal(features['ybar'], 0), tf.equal(features['sensitive'], 1))
        accepted = tf.gather_nd(tf.cast(pred_mean < 0.5, tf.float32), tf.where(test_for_ybar0_s1))
        return self._return_and_store(self.mean(accepted))


class BaseOddsYYbar0S0(Mae):
    """Opportunity P(y=1|s,ybar=1), group 1"""
    display_name = "Base_opportunity_s0"

    def update(self, features, labels, pred_mean):
        test_for_ybar0_s0 = tf.logical_and(tf.equal(features['ybar'], 0), tf.equal(features['sensitive'], 0))
        accepted = tf.gather_nd(1 - labels, tf.where(test_for_ybar0_s0))
        return self._return_and_store(self.mean(accepted))


class BaseOddsYYbar0S1(Mae):
    """Opportunity P(y=1|s,ybar=1), group 2"""
    display_name = "Base_opportunity_s1"

    def update(self, features, labels, pred_mean):
        test_for_ybar0_s1 = tf.logical_and(tf.equal(features['ybar'], 0), tf.equal(features['sensitive'], 1))
        accepted = tf.gather_nd(1 - labels, tf.where(test_for_ybar0_s1))
        return self._return_and_store(self.mean(accepted))


# This is the mapping from string to metric class that is used to find a metric based on the metric flag. Unfortunately,
# this has to be at the end of the file because only here the metric classes have been defined.
MAPPING = {
    'rmse': Rmse,
    'mae': Mae,
    'soft_accuracy': SoftAccuracy,
    'logistic_accuracy': LogisticAccuracy,
    'logistic_accuracy_ybar': LogisticAccuracyYbar,
    'pred_rate_y1_s0': PredictionRateY1S0,
    'pred_rate_y1_s1': PredictionRateY1S1,
    'base_rate_y1_s0': BaseRateY1S0,
    'base_rate_y1_s1': BaseRateY1S1,
    'pred_odds_yybar1_s0': PredictionOddsYYbar1S0,
    'pred_odds_yybar1_s1': PredictionOddsYYbar1S1,
    'base_odds_yybar1_s0': BaseOddsYYbar1S0,
    'base_odds_yybar1_s1': BaseOddsYYbar1S1,
    'pred_odds_yybar0_s0': PredictionOddsYYbar0S0,
    'pred_odds_yybar0_s1': PredictionOddsYYbar0S1,
    'base_odds_yybar0_s0': BaseOddsYYbar0S0,
    'base_odds_yybar0_s1': BaseOddsYYbar0S1,
}

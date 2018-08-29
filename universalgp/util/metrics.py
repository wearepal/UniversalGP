"""Defines methods for metrics"""

import numpy as np
import tensorflow as tf
from tensorflow import manip as tft
from tensorflow import math as tfm
import tensorflow.contrib.eager as tfe
from .. import util

# TODO: split this file into several files


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
        return metrics  # No metric names given -> return empty dictionary

    # First, find all metrics
    dict_of_metrics = {}
    for class_name in dir(util.metrics):  # Loop over everything that is defined in `metrics`
        class_ = getattr(util.metrics, class_name)
        # Here, we filter out all functions and other classes which are not metrics
        if isinstance(class_, type(Metric)) and issubclass(class_, Metric):
            dict_of_metrics[class_.name] = class_

    if isinstance(metric_flag, list):
        metric_list = metric_flag  # `metric_flag` is already a list
    else:
        metric_list = metric_flag.split(',')  # Split `metric_flag` into a list
    for name in metric_list:
        try:
            # Now we can just look up the metrics in the dictionary we created
            metric = dict_of_metrics[name]
        except KeyError:  # No metric found with the name `name`
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
        if update_op is not None:  # `update_op` is only not `None` in graph mode
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
    name = "empty_metric"

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
        """In graph mode stores the result in `self.result` and returns the op so that it can be
        updated

        Does currently nothing in eager mode.
        """
        if not self.is_eager:
            self.result = metric_op[1]
            return metric_op


class Rmse(Metric):
    """Root mean squared error"""
    name = "RMSE"

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
            print(f"{self.name}: {np.sqrt(self.metric.result())}")
        else:
            tf.summary.scalar(self.name, self.result)


class Mae(Metric):
    """Mean absolute error"""
    name = "MAE"

    def __init__(self, is_eager):
        super().__init__(is_eager)
        self.mean = tfe.metrics.Mean() if is_eager else tf.metrics.mean

    def update(self, features, labels, pred_mean):
        return self._return_and_store(self.mean(tf.abs(pred_mean - labels)))

    def record(self):
        if self.is_eager:
            print(f"{self.name}: {self.mean.result()}")
        else:
            tf.summary.scalar(self.name, self.result)


class SoftAccuracy(Metric):
    """Accuracy for softmax output"""
    name = "soft_accuracy"

    def __init__(self, is_eager):
        super().__init__(is_eager)
        self.accuracy = tfe.metrics.Accuracy() if is_eager else tf.metrics.accuracy

    def update(self, features, labels, pred_mean):
        return self._return_and_store(self.accuracy(tf.argmax(labels, axis=1),
                                                    tf.argmax(pred_mean, axis=1)))

    def record(self):
        if self.is_eager:
            print(f"{self.name}: {self.accuracy.result()}")
        else:
            tf.summary.scalar(self.name, self.result)


class LogisticAccuracy(SoftAccuracy):
    """Accuracy for output from the logistic function"""
    name = "logistic_accuracy"

    def update(self, features, labels, pred_mean):
        return self._return_and_store(self.accuracy(tf.cast(labels, tf.int32),
                                                    tf.cast(pred_mean > 0.5, tf.int32)))


class LogisticAccuracyYbar(SoftAccuracy):
    """Accuracy for output from the logistic function"""
    name = "logistic_accuracy_ybar"

    def update(self, features, labels, pred_mean):
        return self._return_and_store(self.accuracy(features['ybar'],
                                                    tf.cast(pred_mean > 0.5, tf.float32)))


class PredictionRateY1S0(Mae):
    """Acceptance Rate, group 1"""
    name = "pred_rate_y1_s0"

    def update(self, features, labels, pred_mean):
        accepted = tft.gather_nd(tf.cast(pred_mean > 0.5, tf.float32),
                                 tf.where(tfm.equal(features['sensitive'], 0)))
        return self._return_and_store(self.mean(accepted))


class PredictionRateY1S1(Mae):
    """Acceptance Rate, group 2"""
    name = "pred_rate_y1_s1"

    def update(self, features, labels, pred_mean):
        accepted = tft.gather_nd(tf.cast(pred_mean > 0.5, tf.float32),
                                 tf.where(tfm.equal(features['sensitive'], 1)))
        return self._return_and_store(self.mean(accepted))


class BaseRateY1S0(Mae):
    """Base acceptance rate, group 1"""
    name = "base_rate_y1_s0"

    def update(self, features, labels, pred_mean):
        accepted = tft.gather_nd(labels, tf.where(tfm.equal(features['sensitive'], 0)))
        return self._return_and_store(self.mean(accepted))


class BaseRateY1S1(Mae):
    """Base acceptance rate, group 2"""
    name = "base_rate_y1_s1"

    def update(self, features, labels, pred_mean):
        accepted = tft.gather_nd(labels, tf.where(tfm.equal(features['sensitive'], 1)))
        return self._return_and_store(self.mean(accepted))


class PredictionOddsYYbar1S0(Mae):
    """Opportunity P(yhat=1|s,ybar=1), group 1"""
    name = "pred_odds_yybar1_s0"

    def update(self, features, labels, pred_mean):
        test_for_ybar1_s0 = tfm.logical_and(tfm.equal(features['ybar'], 1),
                                            tfm.equal(features['sensitive'], 0))
        accepted = tft.gather_nd(tf.cast(pred_mean > 0.5, tf.float32), tf.where(test_for_ybar1_s0))
        return self._return_and_store(self.mean(accepted))


class PredictionOddsYYbar1S1(Mae):
    """Opportunity P(yhat=1|s,ybar=1), group 2"""
    name = "pred_odds_yybar1_s1"

    def update(self, features, labels, pred_mean):
        test_for_ybar1_s1 = tfm.logical_and(tfm.equal(features['ybar'], 1),
                                            tfm.equal(features['sensitive'], 1))
        accepted = tft.gather_nd(tf.cast(pred_mean > 0.5, tf.float32), tf.where(test_for_ybar1_s1))
        return self._return_and_store(self.mean(accepted))


class BaseOddsYYbar1S0(Mae):
    """Opportunity P(y=1|s,ybar=1), group 1"""
    name = "base_odds_yybar1_s0"

    def update(self, features, labels, pred_mean):
        test_for_ybar1_s0 = tfm.logical_and(tfm.equal(features['ybar'], 1),
                                            tfm.equal(features['sensitive'], 0))
        accepted = tft.gather_nd(labels, tf.where(test_for_ybar1_s0))
        return self._return_and_store(self.mean(accepted))


class BaseOddsYYbar1S1(Mae):
    """Opportunity P(y=1|s,ybar=1), group 2"""
    name = "base_odds_yybar1_s1"

    def update(self, features, labels, pred_mean):
        test_for_ybar1_s1 = tfm.logical_and(tfm.equal(features['ybar'], 1),
                                            tfm.equal(features['sensitive'], 1))
        accepted = tft.gather_nd(labels, tf.where(test_for_ybar1_s1))
        return self._return_and_store(self.mean(accepted))


class PredictionOddsYYbar0S0(Mae):
    """Opportunity P(yhat=1|s,ybar=1), group 1"""
    name = "pred_odds_yybar0_s0"

    def update(self, features, labels, pred_mean):
        test_for_ybar0_s0 = tfm.logical_and(tfm.equal(features['ybar'], 0),
                                            tfm.equal(features['sensitive'], 0))
        accepted = tft.gather_nd(tf.cast(pred_mean < 0.5, tf.float32), tf.where(test_for_ybar0_s0))
        return self._return_and_store(self.mean(accepted))


class PredictionOddsYYbar0S1(Mae):
    """Opportunity P(yhat=1|s,ybar=1), group 2"""
    name = "pred_odds_yybar0_s1"

    def update(self, features, labels, pred_mean):
        test_for_ybar0_s1 = tfm.logical_and(tfm.equal(features['ybar'], 0),
                                            tfm.equal(features['sensitive'], 1))
        accepted = tft.gather_nd(tf.cast(pred_mean < 0.5, tf.float32), tf.where(test_for_ybar0_s1))
        return self._return_and_store(self.mean(accepted))


class BaseOddsYYbar0S0(Mae):
    """Opportunity P(y=1|s,ybar=1), group 1"""
    name = "base_odds_yybar0_s0"

    def update(self, features, labels, pred_mean):
        test_for_ybar0_s0 = tfm.logical_and(tfm.equal(features['ybar'], 0),
                                            tfm.equal(features['sensitive'], 0))
        accepted = tft.gather_nd(1 - labels, tf.where(test_for_ybar0_s0))
        return self._return_and_store(self.mean(accepted))


class BaseOddsYYbar0S1(Mae):
    """Opportunity P(y=1|s,ybar=1), group 2"""
    name = "base_odds_yybar0_s1"

    def update(self, features, labels, pred_mean):
        test_for_ybar0_s1 = tfm.logical_and(tfm.equal(features['ybar'], 0),
                                            tfm.equal(features['sensitive'], 1))
        accepted = tft.gather_nd(1 - labels, tf.where(test_for_ybar0_s1))
        return self._return_and_store(self.mean(accepted))


class PredictionOddsYhatY1S0(Mae):
    """Opportunity P(yhat=1|s,y=1), group 1"""
    name = "pred_odds_yhaty1_s0"

    def update(self, features, labels, pred_mean):
        test_for_y1_s0 = tfm.logical_and(tfm.equal(labels, 1), tfm.equal(features['sensitive'], 0))
        accepted = tft.gather_nd(tf.cast(pred_mean > 0.5, tf.float32), tf.where(test_for_y1_s0))
        return self._return_and_store(self.mean(accepted))


class PredictionOddsYhatY1S1(Mae):
    """Opportunity P(yhat=1|s,y=1), group 2"""
    name = "pred_odds_yhaty1_s1"

    def update(self, features, labels, pred_mean):
        test_for_y1_s1 = tfm.logical_and(tfm.equal(labels, 1), tfm.equal(features['sensitive'], 1))
        accepted = tft.gather_nd(tf.cast(pred_mean > 0.5, tf.float32), tf.where(test_for_y1_s1))
        return self._return_and_store(self.mean(accepted))


class PredictionOddsYhatY0S0(Mae):
    """Opportunity P(yhat=0|s,y=0), group 1"""
    name = "pred_odds_yhaty0_s0"

    def update(self, features, labels, pred_mean):
        test_for_y0_s0 = tfm.logical_and(tfm.equal(labels, 0), tfm.equal(features['sensitive'], 0))
        accepted = tft.gather_nd(tf.cast(pred_mean < 0.5, tf.float32), tf.where(test_for_y0_s0))
        return self._return_and_store(self.mean(accepted))


class PredictionOddsYhatY0S1(Mae):
    """Opportunity P(yhat=0|s,y=0), group 2"""
    name = "pred_odds_yhaty0_s1"

    def update(self, features, labels, pred_mean):
        test_for_y0_s1 = tfm.logical_and(tfm.equal(labels, 0), tfm.equal(features['sensitive'], 1))
        accepted = tft.gather_nd(tf.cast(pred_mean < 0.5, tf.float32), tf.where(test_for_y0_s1))
        return self._return_and_store(self.mean(accepted))


def mask_for(features, **kwargs):
    """Create a 'mask' that filters for certain values

    Args:
        features: a dictionary of tensors
        **kwargs: entries of the dictionary with the values, only the first two are used
    Returns:
        a mask
    """
    entries = list(kwargs.items())
    return tf.where(tfm.logical_and(tfm.equal(features[entries[0][0]], entries[0][1]),
                                    tfm.equal(features[entries[1][0]], entries[1][1])))

"""Defines methods for metrics"""

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe


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


class Rmse(Metric):
    """Root mean squared error"""
    def __init__(self, is_eager):
        super().__init__(is_eager)
        self.metric = tfe.metrics.Mean() if is_eager else None

    def update(self, features, labels, pred_mean):
        if self.is_eager:
            self.metric((pred_mean - labels)**2)
        else:
            self.metric = tf.metrics.root_mean_squared_error(labels, pred_mean)
            return self.metric

    def record(self):
        if self.is_eager:
            print(f"RMSE: {np.sqrt(self.metric.result())}")
        else:
            tf.summary.scalar('RMSE', self.metric[0])


class SoftAccuracy(Metric):
    """Accuracy for softmax output"""
    def __init__(self, is_eager):
        super().__init__(is_eager)
        self.metric = tfe.metrics.Accuracy() if is_eager else None

    def update(self, features, labels, pred_mean):
        argmax = [tf.argmax(labels, axis=1), tf.argmax(pred_mean, axis=1)]
        if self.is_eager:
            self.metric(*argmax)
        else:
            self.metric = tf.metrics.accuracy(*argmax)
            return self.metric

    def record(self):
        if self.is_eager:
            print(f"Accuracy: {self.metric.result()}")
        else:
            tf.summary.scalar('Accuracy', self.metric[0])


class LogisticAccuracy(SoftAccuracy):
    """Accuracy for output from the logistic function"""
    def update(self, features, labels, pred_mean):
        cast = [tf.cast(labels, tf.int32), tf.cast(pred_mean > 0.5, tf.int32)]
        if self.is_eager:
            self.metric(*cast)
        else:
            self.metric = tf.metrics.accuracy(*cast)
            return self.metric


# This is the mapping from string to metric class that is used to find a metric based on the metric flag. Unfortunately,
# this has to be at the end of the file because only here the metric classes have been defined.
MAPPING = {
    'rmse': Rmse,
    'soft_accuracy': SoftAccuracy,
    'logistic_accuracy': LogisticAccuracy,
    }

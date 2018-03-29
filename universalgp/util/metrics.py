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
    metric_names = metric_flag.split(',')

    if 'rmse' in metric_names:
        metrics['rmse'] = tfe.metrics.Mean() if is_eager else None

    if 'soft_accuracy' in metric_names:
        metrics['soft_accuracy'] = tfe.metrics.Accuracy() if is_eager else None

    if 'logistic_accuracy' in metric_names:
        metrics['logistic_accuracy'] = tfe.metrics.Accuracy() if is_eager else None

    if not metrics:
        raise ValueError(f"Unknown metric \"{metric_flag}\"")
    return metrics


def update_metrics(metrics, features, labels, pred_mean, is_eager):
    """Update metrics

    Args:
        metrics: a dictionary with the initialized metrics
        features: the input
        labels: the correct labels
        pred_mean: the predicted mean
        is_eager: True if in eager execution
    """
    if 'rmse' in metrics:
        if is_eager:
            metrics['rmse']((pred_mean - labels)**2)
        else:
            metrics['rmse'] = tf.metrics.root_mean_squared_error(labels, pred_mean)

    if 'soft_accuracy' in metrics:
        argmax = [tf.argmax(pred_mean, axis=1), tf.argmax(labels, axis=1)]
        if is_eager:
            metrics['soft_accuracy'](*argmax)
        else:
            metrics['soft_accuracy'] = tf.metrics.accuracy(*argmax)

    if 'logistic_accuracy' in metrics:
        cast = [tf.cast(pred_mean > 0.5, tf.int32), tf.cast(labels, tf.int32)]
        if is_eager:
            metrics['logistic_accuracy'](*cast)
        else:
            metrics['logistic_accuracy'] = tf.metrics.accuracy(*cast)


def record_metrics(metrics, is_eager):
    """Print the result or record it in the summary

    Args:
        metrics: a dictionary with the updated metrics
        is_eager: True if in eager execution
    """
    if 'rmse' in metrics:
        if is_eager:
            print(f"RMSE: {np.sqrt(metrics['rmse'].result())}")
        else:
            tf.summary.scalar('RMSE', metrics['rmse'][0])

    if 'soft_accuracy' in metrics:
        if is_eager:
            print(f"Accuracy: {metrics['soft_accuracy'].result()}")
        else:
            tf.summary.scalar('Accuracy', metrics['soft_accuracy'][0])

    if 'logistic_accuracy' in metrics:
        if is_eager:
            print(f"Accuracy: {metrics['logistic_accuracy'].result()}")
        else:
            tf.summary.scalar('Accuracy', metrics['logistic_accuracy'][0])

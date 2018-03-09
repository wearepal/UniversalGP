#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is mainly for defining flags and choosing the right dataset.
"""
import sys
import tensorflow as tf

import datasets
import universalgp

FLAGS = tf.app.flags.FLAGS
### GP flags
tf.app.flags.DEFINE_string('tf_mode', 'eager', 'The mode in which Tensorflow is run. Either `graph` or `eager`.')
tf.app.flags.DEFINE_string('data', 'simple_example', 'Dataset to use')
# tf.app.flags.DEFINE_string('data', 'mnist', 'Dataset to use')
tf.app.flags.DEFINE_string('inf', 'Variational', 'Inference method')
# tf.app.flags.DEFINE_string('inf', 'Exact', 'Inference method')
tf.app.flags.DEFINE_string('lik', 'LikelihoodGaussian', 'Likelihood function')
# tf.app.flags.DEFINE_string('lik', 'LikelihoodSoftmax', 'Likelihood function')
tf.app.flags.DEFINE_string('cov', 'SquaredExponential', 'Covariance function')
tf.app.flags.DEFINE_float('lr', 0.005, 'Learning rate')
tf.app.flags.DEFINE_boolean('use_ard', True, 'Whether to use an automatic relevance determination kernel')
tf.app.flags.DEFINE_float('length_scale', 1.0, 'Initial lenght scale for the kernel')
tf.app.flags.DEFINE_string('metric', 'rmse', 'metric for evaluating the trained model')
tf.app.flags.DEFINE_integer('loo_steps', None, 'Number of steps for optimizing LOO loss')
### Tensorflow flags
tf.app.flags.DEFINE_string('model_name', 'local', 'Name of model (used for name of checkpoints)')
tf.app.flags.DEFINE_integer('batch_size', 50, 'Batch size')
tf.app.flags.DEFINE_integer('train_steps', 500, 'Number of training steps')
tf.app.flags.DEFINE_integer('eval_epochs', 10000, 'Number of epochs between evaluations')
tf.app.flags.DEFINE_integer('summary_steps', 100, 'How many steps between saving summary')
tf.app.flags.DEFINE_integer('chkpnt_steps', 5000, 'How many steps between saving checkpoints')
tf.app.flags.DEFINE_string('save_dir', None,  # '/its/home/tk324/tensorflow/',
                           'Directory where the checkpoints and summaries are saved')
tf.app.flags.DEFINE_boolean('plot', True, 'Whether to plot the result')
tf.app.flags.DEFINE_integer('logging_steps', 1, 'How many steps between logging the loss')
tf.app.flags.DEFINE_string('gpus', '0', 'Which GPUs to use (should normally only be one)')
tf.app.flags.DEFINE_boolean('save_vars', False, 'Whether to save the trained variables as numpy arrays in the end')


def main(_):
    """The main entry point

    This functions constructs the data set and then calls the requested training loop.
    """
    if FLAGS.tf_mode == 'graph':
        tf.logging.set_verbosity(tf.logging.INFO)
        train_func = universalgp.train_graph
    if FLAGS.tf_mode == 'eager':
        train_func = universalgp.train_eager
    else:
        ValueError('Unknown tf_mode: "{}"'.format(FLAGS.tf_mode))
    dataset = getattr(datasets, FLAGS.data)()
    train_func.gp(dataset)


if __name__ == '__main__':
    tf.app.run(main=main, argv=sys.argv)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is mainly for defining flags and choosing the right dataset.
"""
import sys
import tensorflow as tf

from universalgp import train_eager, datasets

FLAGS = tf.compat.v1.app.flags.FLAGS

# ---GP flags---
# tf.compat.v1.app.flags.DEFINE_string('data', 'simple_example', 'Dataset to use')
tf.compat.v1.app.flags.DEFINE_string('data', 'sensitive_odds_example', 'Dataset to use')
# tf.compat.v1.app.flags.DEFINE_string('data', 'mnist', 'Dataset to use')
tf.compat.v1.app.flags.DEFINE_string('inf', 'Variational', 'Inference method')
# tf.compat.v1.app.flags.DEFINE_string('inf', 'Exact', 'Inference method')
tf.compat.v1.app.flags.DEFINE_string('cov', 'SquaredExponential', 'Covariance function')
tf.compat.v1.app.flags.DEFINE_float('lr', 0.005, 'Learning rate')
tf.compat.v1.app.flags.DEFINE_integer('loo_steps', 0,
                                      'Number of steps for optimizing LOO loss; 0 disables')
tf.compat.v1.app.flags.DEFINE_integer(
    'nelbo_steps', 0, 'Number of steps for optimizing NELBO loss; 0 means same as loo_steps')
tf.compat.v1.app.flags.DEFINE_integer(
    'num_all', 200, 'Suggested total number of examples (datasets don\'t have to use it)')
tf.compat.v1.app.flags.DEFINE_integer(
    'num_train', 50, 'Suggested number of train examples (datasets don\'t have to use it)')
tf.compat.v1.app.flags.DEFINE_integer(
    'num_inducing', 50, 'Suggested number of inducing inputs (datasets don\'t have to use it)')

# ---Tensorflow flags---
tf.compat.v1.app.flags.DEFINE_string('optimizer', 'RMSprop', 'Optimizer to use for SGD')
tf.compat.v1.app.flags.DEFINE_string('model_name', 'local',
                                     'Name of model (used for name of checkpoints)')
tf.compat.v1.app.flags.DEFINE_integer('batch_size', 50, 'Batch size')
tf.compat.v1.app.flags.DEFINE_integer('train_steps', 500, 'Number of training steps')
tf.compat.v1.app.flags.DEFINE_integer('eval_epochs', 10000, 'Number of epochs between evaluations')
tf.compat.v1.app.flags.DEFINE_integer('summary_steps', 100, 'How many steps between saving summary')
tf.compat.v1.app.flags.DEFINE_integer('chkpnt_steps', 5000,
                                      'How many steps between saving checkpoints')
tf.compat.v1.app.flags.DEFINE_string(
    'save_dir', '', 'Directory where the checkpoints and summaries are saved (or \'\')')
tf.compat.v1.app.flags.DEFINE_string('plot', '', 'Which function to use for plotting (or \'\')')
tf.compat.v1.app.flags.DEFINE_integer('logging_steps', 1, 'How many steps between logging the loss')
tf.compat.v1.app.flags.DEFINE_string('gpus', '0', 'Which GPUs to use (should normally only be one)')
tf.compat.v1.app.flags.DEFINE_string(
    'preds_path', '', 'Path where the predictions for the test data will be save (or "")')
tf.compat.v1.app.flags.DEFINE_integer('eval_throttle', 600,
                                      'How long to wait before evaluating in seconds')
tf.compat.v1.app.flags.DEFINE_integer('lr_drop_steps', 0,
                                      'Number of steps before doing a learning rate drop')
tf.compat.v1.app.flags.DEFINE_float('lr_drop_factor', 0.2,
                                    'For learning rate drop multiply by this factor')

# you can specify a flag file here where you can put your flags instead of passing them from the
# command line
# FLAGFILE = "scripts/flagfiles/performance_test.sh"
FLAGFILE = ""


def main(_):
    """
    The main entry point

    This functions constructs the data set and then calls the requested training loop.
    """
    args = {flag: getattr(FLAGS, flag) for flag in FLAGS}  # convert FLAGS to dictionary
    # take dataset function from the module `datasets` and execute it
    dataset = getattr(datasets, FLAGS.data)(args)
    train_eager.train_gp(dataset, args)


if __name__ == '__main__':
    tf.compat.v1.app.run(main=main,
                         argv=[sys.argv[0], f"--flagfile={FLAGFILE}"] if FLAGFILE else sys.argv)

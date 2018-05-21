#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is mainly for defining flags and choosing the right dataset.
"""
import sys
import tensorflow as tf

from universalgp import train_graph, train_eager, datasets

FLAGS = tf.app.flags.FLAGS

# ---GP flags---
tf.app.flags.DEFINE_string('tf_mode', 'graph',
                           'The mode in which Tensorflow is run. Either `graph` or `eager`.')
# tf.app.flags.DEFINE_string('data', 'simple_example', 'Dataset to use')
tf.app.flags.DEFINE_string('data', 'sensitive_odds_example', 'Dataset to use')
# tf.app.flags.DEFINE_string('data', 'mnist', 'Dataset to use')
tf.app.flags.DEFINE_string('inf', 'Variational', 'Inference method')
# tf.app.flags.DEFINE_string('inf', 'Exact', 'Inference method')
tf.app.flags.DEFINE_string('cov', 'SquaredExponential', 'Covariance function')
tf.app.flags.DEFINE_float('lr', 0.005, 'Learning rate')
tf.app.flags.DEFINE_integer('loo_steps', None, 'Number of steps for optimizing LOO loss')
tf.app.flags.DEFINE_integer('num_all', 200,
                            'Suggested total number of examples (datasets don\'t have to use it)')
tf.app.flags.DEFINE_integer('num_train', 50,
                            'Suggested number of train examples (datasets don\'t have to use it)')
tf.app.flags.DEFINE_integer('num_inducing', 50,
                            'Suggested number of inducing inputs (datasets don\'t have to use it)')

# ---Tensorflow flags---
tf.app.flags.DEFINE_string('optimizer', 'RMSPropOptimizer', 'Optimizer to use for SGD')
tf.app.flags.DEFINE_string('model_name', 'local', 'Name of model (used for name of checkpoints)')
tf.app.flags.DEFINE_integer('batch_size', 50, 'Batch size')
tf.app.flags.DEFINE_integer('train_steps', 500, 'Number of training steps')
tf.app.flags.DEFINE_integer('eval_epochs', 10000, 'Number of epochs between evaluations')
tf.app.flags.DEFINE_integer('summary_steps', 100, 'How many steps between saving summary')
tf.app.flags.DEFINE_integer('chkpnt_steps', 5000, 'How many steps between saving checkpoints')
tf.app.flags.DEFINE_string('save_dir', None,  # '/its/home/tk324/tensorflow/',
                           'Directory where the checkpoints and summaries are saved')
tf.app.flags.DEFINE_string('plot', None, 'Which function to use for plotting (or None)')
tf.app.flags.DEFINE_integer('logging_steps', 1, 'How many steps between logging the loss')
tf.app.flags.DEFINE_string('gpus', '0', 'Which GPUs to use (should normally only be one)')
tf.app.flags.DEFINE_string('preds_path', '',
                           'Path where the predictions for the test data will be save (or "")')
tf.app.flags.DEFINE_integer('eval_throttle', 600, 'How long to wait before evaluating in seconds')

# you can specify a flag file here where you can put your flags instead of passing them from the
# command line
FLAGFILE = ""  # "scripts/flagfiles/simple_example.sh"


def main(_):
    """
    The main entry point

    This functions constructs the data set and then calls the requested training loop.
    """
    if FLAGS.tf_mode == 'graph':
        train_func = train_graph
        tf.logging.set_verbosity(tf.logging.INFO)  # print logging information (e.g. the loss)
    elif FLAGS.tf_mode == 'eager':
        train_func = train_eager
        tf.enable_eager_execution()  # enable Eager Execution (tensors are evaluated immediately)
    else:
        raise ValueError(f"Unknown tf_mode: \"{FLAGS.tf_mode}\"")
    args = {flag: getattr(FLAGS, flag) for flag in FLAGS}  # convert FLAGS to dictionary
    # take dataset function from the module `datasets` and execute it
    dataset = getattr(datasets, FLAGS.data)(args)
    train_func.train_gp(dataset, args)


if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0], f"--flagfile={FLAGFILE}"] if FLAGFILE else sys.argv)

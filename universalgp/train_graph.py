"""
Main training loop for a Gaussian Process based on a Tensorflow graph
"""
from pathlib import Path
import tensorflow as tf
import numpy as np

from . import inf, cov, lik, util

FLAGS = tf.app.flags.FLAGS


def build_gaussian_process(features, labels, mode, params: dict):
    """Create the Gaussian Process

    This function is called 3 times to create 3 different graphs which share some variables. It is called with
    mode == TRAIN, mode == EVAL and mode == PREDICT.

    Args:
        features: the inputs (has to be decoded with `tf.feature_column.input_layer()`)
        labels: the outputs
        mode: TRAIN, EVAL or PREDICT
        params: a dictionary of parameters
    Returns:
        a `tf.EstimatorSpec`
    """
    # Recover inputs
    inputs = tf.feature_column.input_layer(features, params['feature_columns'])

    # Gather parameters
    cov_func = [getattr(cov, FLAGS.cov)(params['input_dim'], FLAGS.length_scale, iso=not FLAGS.use_ard)
                for _ in range(params['output_dim'])]
    lik_func = getattr(lik, FLAGS.lik)()
    if mode == tf.estimator.ModeKeys.TRAIN:
        inducing_param = params['inducing_inputs']
    else:  # when we're not training, we only need the shape of the inducing inputs
        inducing_param = params['inducing_inputs'].shape[-2]

    # Initialize GP
    inf_func = getattr(inf, FLAGS.inf)(cov_func, lik_func, params['num_train'], inducing_param)

    pred_mean, pred_var = inf_func.predict(inputs)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions={'mean': pred_mean, 'var': pred_var})

    # Do inference
    obj_func, inf_param = inf_func.inference(inputs, labels, mode == tf.estimator.ModeKeys.TRAIN)
    loss = sum(obj_func.values())
    # Get hyper parameters
    hyper_params = lik_func.get_params() + sum([k.get_params() for k in cov_func], [])

    # Compute evaluation metrics.
    if FLAGS.metric == 'rmse':
        rmse = tf.metrics.root_mean_squared_error(labels, pred_mean, name='rmse_op')
        metrics = {'RMSE': rmse}
        tf.summary.scalar('RMSE', rmse[0])
    elif FLAGS.metric == 'accuracy':
        acc = tf.metrics.accuracy(tf.argmax(labels, axis=1), tf.argmax(pred_mean, axis=1))
        metrics = {'accuracy': acc}
        tf.summary.scalar('accuracy', acc[0])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.RMSPropOptimizer(learning_rate=FLAGS.lr)
    if FLAGS.loo_steps is not None:
        # Alternate the loss function
        global_step = tf.train.get_global_step()
        mask = tf.equal((global_step // FLAGS.loo_steps) % 2, 0)
        nelbo_loss = tf.where(mask, obj_func['NELBO'], 0.0)
        loo_loss = tf.where(mask, 0.0, obj_func['LOO_VARIATIONAL'])
        train_nelbo = optimizer.minimize(nelbo_loss, global_step=global_step, var_list=inf_param + hyper_params)
        train_loo = optimizer.minimize(loo_loss, global_step=global_step, var_list=hyper_params)
        train_op = tf.group(train_nelbo, train_loo)
    else:
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step(), var_list=inf_param + hyper_params)
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[
        tf.train.LoggingTensorHook(obj_func, every_n_iter=FLAGS.logging_steps)])


def train_gp(data):
    """The main entry point

    This functions calls other functions as necessary to construct a graph and then runs the training loop.
    """

    # Feature columns describe how to use the input.
    my_feature_columns = [tf.feature_column.numeric_column(key='input', shape=data.input_dim)]

    gp = tf.estimator.Estimator(
        model_fn=build_gaussian_process,
        params={
            'feature_columns': my_feature_columns,
            'input_dim': data.input_dim,
            'output_dim': data.output_dim,
            'num_train': data.num_train,
            'inducing_inputs': data.inducing_inputs,
        },
        model_dir=None if FLAGS.save_dir is None else str(Path(FLAGS.save_dir) / Path(FLAGS.model_name)),
        config=tf.estimator.RunConfig().replace(
            save_checkpoints_secs=None,
            save_checkpoints_steps=FLAGS.chkpnt_steps,
            save_summary_steps=FLAGS.summary_steps,
            keep_checkpoint_max=5,
            log_step_count_steps=FLAGS.chkpnt_steps,
            session_config=tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list=FLAGS.gpus))))

    # Settings for training
    trainer = tf.estimator.TrainSpec(input_fn=lambda: data.train_fn().repeat(FLAGS.eval_epochs).batch(FLAGS.batch_size),
                                     max_steps=FLAGS.train_steps)

    # Settings for evaluation
    evaluator = tf.estimator.EvalSpec(input_fn=lambda: data.test_fn().batch(FLAGS.batch_size))

    tf.estimator.train_and_evaluate(gp, trainer, evaluator)  # this can be replaced by a loop that calls gp.train()

    if FLAGS.save_vars and FLAGS.save_dir is not None:
        print("Saving variables...")
        var_collection = {name: gp.get_variable_value(name) for name in gp.get_variable_names()}
        np.savez_compressed(Path(FLAGS.save_dir) / Path(FLAGS.model_name) / Path("vars"), **var_collection)
    if FLAGS.plot:
        # Create predictions
        predictions_gen = gp.predict(input_fn=lambda: data.test_fn().batch(len(data.xtest)))
        pred_mean = []
        pred_var = []
        for prediction in predictions_gen:
            pred_mean.append(prediction['mean'])
            pred_var.append(prediction['var'])
        pred_mean = np.stack(pred_mean)
        pred_var = np.stack(pred_var)
        util.simple_1d(pred_mean, pred_var, data.xtrain, data.ytrain, data.xtest, data.ytest)
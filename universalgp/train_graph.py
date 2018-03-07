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
    inducing_inputs = params.get('inducing_inputs', None)

    # Construct graph
    if FLAGS.inf == 'Variational':
        inf_func = inf.Variational(cov_func, lik_func, FLAGS.diag_post, FLAGS.num_components, FLAGS.num_samples,
                                   FLAGS.optimize_inducing, FLAGS.loo)
        labels = tf.constant(0.) if labels is None else labels
        obj_func, preds, inf_param = inf_func.inference(inputs, labels, inputs, params['num_train'], inducing_inputs)
    elif FLAGS.inf == 'Exact' or FLAGS.inf == 'Loo':
        inf_func = getattr(inf, FLAGS.inf)(cov_func, lik_func)
        # TODO: the following should be moved into the inference class
        train_inputs = tf.get_variable('saved_train_inputs', [params['num_train'], inputs.shape[1]], trainable=False)
        train_outputs = tf.get_variable('saved_train_outputs', [params['num_train'], 1], trainable=False)
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_inputs = train_inputs.assign(inputs)
            train_outputs = train_outputs.assign(labels)
        elif mode == tf.estimator.ModeKeys.EVAL:
            train_inputs, train_outputs = inputs, labels
        obj_func, preds, inf_param = inf_func.inference(train_inputs, train_outputs, inputs, params['num_train'], None)

    loss = sum(obj_func.values())
    pred_mean, pred_var = preds

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Only do prediction
        return tf.estimator.EstimatorSpec(mode, predictions={'mean': pred_mean, 'var': pred_var})

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
    # if we want to use multiple loss functions, see the following:
    # https://github.com/tensorflow/tensorflow/issues/15773#issuecomment-356451902
    # in order to alternate the loss, the global step has to be taken into account (otherwise we stay on the same batch)
    if FLAGS.loo and FLAGS.loo_steps is not None:
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


def run(data):
    """The main entry point

    This functions calls other functions as necessary to construct a graph and then runs the training loop.
    """

    # Feature columns describe how to use the input.
    my_feature_columns = [tf.feature_column.numeric_column(key='input', shape=data['input_dim'])]

    gp = tf.estimator.Estimator(
        model_fn=build_gaussian_process,
        params={
            'feature_columns': my_feature_columns,
            'input_dim': data['input_dim'],
            'output_dim': data['output_dim'],
            'num_train': data['num_train'],
            'inducing_inputs': data['inducing_inputs'],
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
    trainer = tf.estimator.TrainSpec(lambda: data['train_fn']().repeat(FLAGS.eval_epochs).batch(FLAGS.batch_size),
                                     max_steps=FLAGS.train_steps)

    # Settings for evaluation
    evaluator = tf.estimator.EvalSpec(input_fn=lambda: data['test_fn']().batch(FLAGS.batch_size))

    tf.estimator.train_and_evaluate(gp, trainer, evaluator)  # this can be replaced by a loop that calls gp.train()

    if FLAGS.save_vars and FLAGS.save_dir is not None:
        print("Saving variables...")
        var_collection = {name: gp.get_variable_value(name) for name in gp.get_variable_names()}
        np.savez_compressed(Path(FLAGS.save_dir) / Path(FLAGS.model_name) / Path("vars"), **var_collection)
    if FLAGS.plot:
        # Create predictions
        predictions_gen = gp.predict(input_fn=lambda: data['test_fn']().batch(len(data['xtest'])))
        pred_mean = []
        pred_var = []
        for prediction in predictions_gen:
            pred_mean.append(prediction['mean'])
            pred_var.append(prediction['var'])
        pred_mean = np.stack(pred_mean)
        pred_var = np.stack(pred_var)
        util.simple_1d(pred_mean, pred_var, data['xtrain'], data['ytrain'], data['xtest'], data['ytest'])

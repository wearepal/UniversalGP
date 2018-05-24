"""
Main training loop for a Gaussian Process based on a Tensorflow graph
"""
from pathlib import Path
import tensorflow as tf
import numpy as np

from . import util


def build_gaussian_process(features, labels, mode, params: dict):
    """Create the Gaussian Process

    This function is called 3 times to create 3 different graphs which share some variables. It is
    called with mode == TRAIN, mode == EVAL and mode == PREDICT.

    Args:
        features: a dictionary of the inputs
        labels: the outputs
        mode: TRAIN, EVAL or PREDICT
        params: a dictionary of parameters
    Returns:
        a `tf.EstimatorSpec`
    """
    if mode == tf.estimator.ModeKeys.TRAIN:
        inducing_param = params['inducing_inputs']
    else:
        # not training -> only need shape of the inducing inputs
        inducing_param = params['inducing_inputs'].shape[-2]

    inf_func, hyper_params, optimizer = util.construct_from_flags(
        params, params['input_dim'], params['output_dim'], params['lik'], inducing_param,
        params['num_train'])

    pred_mean, pred_var = inf_func.predict(features)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions={'mean': pred_mean, 'var': pred_var})

    # Do inference
    obj_func, inf_param = inf_func.inference(features, labels, mode == tf.estimator.ModeKeys.TRAIN)

    # Compute evaluation metrics.
    metrics = util.init_metrics(params['metric'], False)
    metric_ops = util.update_metrics(metrics, features, labels, pred_mean)
    util.record_metrics(metrics)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=obj_func['loss'], eval_metric_ops=metric_ops)

    assert mode == tf.estimator.ModeKeys.TRAIN
    global_step = tf.train.get_global_step()
    for_logging = {'step': global_step, **obj_func, **{k: v[1] for k, v in metric_ops.items()}}

    if params['loo_steps']:
        # Alternate the loss function
        mask = tf.equal((global_step // params['loo_steps']) % 2, 0)
        nelbo_loss = tf.where(mask, obj_func['NELBO'], 0.0)
        loo_loss = tf.where(mask, 0.0, obj_func['LOO_VARIATIONAL'])
        train_nelbo = optimizer.minimize(nelbo_loss, global_step=global_step,
                                         var_list=inf_param + hyper_params)
        train_loo = optimizer.minimize(loo_loss, global_step=global_step, var_list=hyper_params)
        train_op = tf.group(train_nelbo, train_loo)
    else:
        train_op = optimizer.minimize(obj_func['loss'],
                                      global_step=global_step, var_list=inf_param + hyper_params)
    return tf.estimator.EstimatorSpec(
        mode, loss=obj_func['loss'], train_op=train_op, training_hooks=[
            tf.train.LoggingTensorHook(for_logging, every_n_iter=params['logging_steps'])])


def train_gp(data, args):
    """Train a GP model and return it. This function uses Tensorflow's Estimator API which
    constructs graphs. This functions calls other functions as necessary to construct a graph and
    then runs the training loop.

    Args:
        dataset: a NamedTuple that contains information about the dataset
        args: parameters in form of a dictionary
    Returns:
        the trained GP as `tf.estimator.Estimator`
    """
    # Get certain parameters from `data`
    params = {param: getattr(data, param) for param in [
        'train_feature_columns', 'test_feature_columns', 'input_dim', 'output_dim', 'num_train',
        'inducing_inputs', 'metric', 'lik']}
    out_dir = str(Path(args['save_dir']) / Path(args['model_name'])) if args['save_dir'] else None
    gp = tf.estimator.Estimator(
        model_fn=build_gaussian_process,
        params={**params, **args},
        model_dir=out_dir,
        config=tf.estimator.RunConfig().replace(
            save_checkpoints_secs=None,
            save_checkpoints_steps=args['chkpnt_steps'],
            save_summary_steps=args['summary_steps'],
            keep_checkpoint_max=5,
            log_step_count_steps=args['chkpnt_steps'],
            session_config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(visible_device_list=args['gpus']))))

    # Settings for training
    trainer = tf.estimator.TrainSpec(
        input_fn=lambda: data.train_fn().shuffle(50_000).repeat(args['eval_epochs'])
        .batch(args['batch_size']),
        max_steps=args['train_steps'])

    # Settings for evaluation
    evaluator = tf.estimator.EvalSpec(input_fn=lambda: data.test_fn().batch(args['batch_size']),
                                      throttle_secs=args['eval_throttle'])

    tf.estimator.train_and_evaluate(gp, trainer, evaluator)  # replaceable by a loop over gp.train()

    if args['plot'] or args['preds_path']:
        print("Making predictions...")
        predictions_gen = gp.predict(input_fn=lambda: data.test_fn().batch(len(data.xtest)))
        predictions = np.array([(p['mean'], p['var']) for p in predictions_gen])
        util.post_training(predictions[:, 0], predictions[:, 1], out_dir, data, args)
    return gp

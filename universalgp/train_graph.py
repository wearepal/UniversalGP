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
    dataset = params['dataset']
    if mode == tf.estimator.ModeKeys.TRAIN:
        inducing_param = dataset.inducing_inputs
    else:
        # not training -> only need shape of the inducing inputs
        inducing_param = dataset.inducing_inputs.shape[-2]

    global_step = tf.train.get_global_step()
    optimizer = util.get_optimizer(params, global_step)
    inf_func, hyper_params = util.construct_from_flags(params, dataset, inducing_param)

    pred_mean, pred_var = inf_func.predict(features)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions={'mean': pred_mean, 'var': pred_var})

    # Do inference
    obj_func, inf_param = inf_func.inference(features, labels, mode == tf.estimator.ModeKeys.TRAIN)

    # Compute evaluation metrics.
    metrics = util.init_metrics(dataset.metric, False)
    metric_ops = util.update_metrics(metrics, features, labels, pred_mean)
    util.record_metrics(metrics)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=obj_func['loss'], eval_metric_ops=metric_ops)

    assert mode == tf.estimator.ModeKeys.TRAIN
    for_logging = {'step': global_step, **obj_func, **{k: v[1] for k, v in metric_ops.items()}}

    if params['loo_steps']:
        # Alternate the loss function between NELBO loss and LOO loss
        nelbo_steps = params['nelbo_steps'] if params['nelbo_steps'] > 0 else params['loo_steps']
        mask = tf.less(global_step % (nelbo_steps + params['loo_steps']), nelbo_steps)
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


def train_gp(dataset, args):
    """Train a GP model and return it. This function uses Tensorflow's Estimator API which
    constructs graphs. This functions calls other functions as necessary to construct a graph and
    then runs the training loop.

    Args:
        dataset: a NamedTuple that contains information about the dataset
        args: parameters in form of a dictionary
    Returns:
        the trained GP as `tf.estimator.Estimator`
    """
    out_dir = str(Path(args['save_dir']) / Path(args['model_name'])) if args['save_dir'] else None
    gp = tf.estimator.Estimator(
        model_fn=build_gaussian_process,
        params={**args, 'dataset': dataset._replace(train_fn=None, test_fn=None)},
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
        input_fn=lambda: dataset.train_fn().shuffle(50_000).repeat(args['eval_epochs'])
        .batch(args['batch_size']),
        max_steps=args['train_steps'])

    # Settings for evaluation
    evaluator = tf.estimator.EvalSpec(input_fn=lambda: dataset.test_fn().batch(args['batch_size']),
                                      throttle_secs=args['eval_throttle'])

    tf.estimator.train_and_evaluate(gp, trainer, evaluator)  # replaceable by a loop over gp.train()

    if args['plot'] or args['preds_path']:
        print("Making predictions...")
        predictions_gen = gp.predict(input_fn=lambda: dataset.test_fn().batch(len(dataset.xtest)))
        predictions = np.array([(p['mean'], p['var']) for p in predictions_gen])
        util.post_training(predictions[:, 0], predictions[:, 1], out_dir, dataset, args)
    return gp

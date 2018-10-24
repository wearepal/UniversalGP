"""Eager training of GP model."""

import time
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from . import util


def fit(gp, optimizer, data, step_counter, args):
    """Trains model on `train_data` using `optimizer`.

    Args:
        gp: gaussian process
        optimizer: tensorflow optimizer
        data: a `tf.data.Dataset` object
        step_counter: variable to keep track of the training step
        args: additional parameters
    """

    start = time.time()
    for (batch_num, (features, outputs)) in enumerate(data):
        # Record the operations used to compute the loss given the input, so that the gradient of
        # the loss with respect to the variables can be computed.
        with tf.GradientTape() as tape:
            obj_func = gp.inference(features, outputs, True)
        # Compute gradients
        all_params = gp.trainable_variables
        if args['loo_steps']:
            # Alternate loss between NELBO and LOO
            nelbo_steps = args['nelbo_steps'] if args['nelbo_steps'] > 0 else args['loo_steps']
            if (step_counter.numpy() % (nelbo_steps + args['loo_steps'])) < nelbo_steps:
                grads_and_params = zip(tape.gradient(obj_func['NELBO'], all_params), all_params)
            else:
                grads_and_params = zip(
                    tape.gradient(obj_func['LOO_VARIATIONAL'], all_params), all_params)
        else:
            grads_and_params = zip(tape.gradient(obj_func['loss'], all_params), all_params)
        # Apply gradients
        optimizer.apply_gradients(grads_and_params, global_step=step_counter)

        if args['logging_steps'] != 0 and batch_num % args['logging_steps'] == 0:
            print(f"Step #{step_counter.numpy()} ({time.time() - start:.4f} sec)\t", end=' ')
            for loss_name, loss_value in obj_func.items():
                print(f"{loss_name}: {loss_value:.2f}", end=' ')
            print("")  # newline
            start = time.time()


def evaluate(gp, data, dataset_metric):
    """Perform an evaluation of `inf_func` on the examples from `dataset`.

    Args:
        gp: gaussian process
        data: a `tf.data.Dataset` object
        dataset_metric: the metrics that are supposed to be evaluated
    """
    avg_loss = tfe.metrics.Mean('loss')
    metrics = util.init_metrics(dataset_metric, True)

    for (features, outputs) in data:
        pred_mean, _ = gp.predict(features)
        obj_func = gp.inference(features, outputs, False)
        avg_loss(obj_func['loss'])
        util.update_metrics(metrics, features, outputs, pred_mean)
    print(f"Test set: Average loss: {avg_loss.result()}")
    util.record_metrics(metrics)


def predict(test_inputs, saved_model, dataset_info, args):
    """Predict outputs given test inputs.

    This function can be called from a different module and should still work.

    Args:
        test_inputs: ndarray. Points on which we wish to make predictions.
            Dimensions: num_test * input_dim.
        saved_model: path to saved model
        dataset_info: info about the dataset
        args: additional parameters

    Returns:
        ndarray. The predicted mean of the test inputs. Dimensions: num_test * output_dim.
        ndarray. The predicted variance of the test inputs. Dimensions: num_test * output_dim.
    """
    if args['batch_size'] is None:
        num_batches = 1
    else:
        num_batches = util.ceil_divide(test_inputs.shape[0], args['batch_size'])
    # num_inducing = dataset_info.inducing_inputs.shape[0]

    gp = util.construct_from_flags(args, dataset_info)
    checkpoint = tf.train.Checkpoint(gp=gp)
    checkpoint.restore(saved_model)

    test_inputs = np.array_split(test_inputs, num_batches)
    mean, var = [0.0] * num_batches, [0.0] * num_batches
    for i in range(num_batches):
        mean[i], var[i] = gp.predict({'input': tf.constant(test_inputs[i], dtype=tf.float32)})

    return np.concatenate(mean, axis=0), np.concatenate(var, axis=0)


def train_gp(dataset, args):
    """Train a GP model and return it. This function uses Tensorflow's eager execution.

    Args:
        dataset: a NamedTuple that contains information about the dataset
        args: parameters in form of a dictionary
    Returns:
        trained GP
    """

    # Set checkpoint path
    if args['save_dir']:
        out_dir = Path(args['save_dir']) / Path(args['model_name'])
        tf.gfile.MakeDirs(str(out_dir))
    else:
        out_dir = Path(mkdtemp())  # Create temporary directory
    checkpoint_prefix = out_dir / Path('model.ckpt')

    # Construct objects
    step_counter = tf.train.get_or_create_global_step()
    gp = util.construct_from_flags(args, dataset)
    optimizer = util.get_optimizer(args, step_counter)

    # Restore from existing checkpoint
    checkpoint = tf.train.Checkpoint(gp=gp, optimizer=optimizer, step_counter=step_counter)
    checkpoint.restore(tf.train.latest_checkpoint(out_dir))

    step = 0
    # shuffle and repeat for the required number of epochs
    train_data = dataset.train_fn().shuffle(50_000).repeat(args['eval_epochs']).batch(
        args['batch_size'])
    # start with one evaluation
    evaluate(gp, dataset.test_fn().batch(args['batch_size']), dataset.metric)
    while step < args['train_steps']:
        start = time.time()
        # take *at most* (train_steps - step) batches so that we don't run longer than `train_steps`
        fit(gp, optimizer, train_data.take(args['train_steps'] - step), step_counter, args)
        end = time.time()
        step = step_counter.numpy()
        print(f"Train time for the last {args['eval_epochs']} epochs (global step {step}):"
              f" {end - start:0.2f}s")
        evaluate(gp, dataset.test_fn().batch(args['batch_size']), dataset.metric)
        # TODO: don't ignore the 'chkpnt_steps' flag
        ckpt_path = checkpoint.save(checkpoint_prefix)
        print(f"Saved checkpoint in '{ckpt_path}'")

    if args['plot'] or args['preds_path']:  # Create predictions
        tf.reset_default_graph()
        mean, var = predict(dataset.xtest, tf.train.latest_checkpoint(out_dir), dataset, args)
        util.post_training(mean, var, out_dir, dataset, args)
    return gp

"""Eager training of GP model."""

import time
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from . import util


def train_gp(dataset, args):
    """Train a GP model and return it. This function uses Tensorflow's eager execution.

    Args:
        dataset: a NamedTuple that contains information about the dataset
        args: parameters in form of a dictionary
    Returns:
        trained GP
    """

    # Select device
    device = '/gpu:' + args['gpus']
    if args['gpus'] is None or tfe.num_gpus() <= 0:
        device = '/cpu:0'
    print('Using device {}'.format(device))

    optimizer = tf.train.RMSPropOptimizer(learning_rate=args['lr'])

    # Set checkpoint path
    if args['save_dir'] is not None:
        out_dir = Path(args['save_dir']) / Path(args['model_name'])
        tf.gfile.MakeDirs(str(out_dir))
    else:
        out_dir = Path(mkdtemp())  # Create temporary directory
    checkpoint_prefix = out_dir / Path('model.ckpt')
    step_counter = tf.train.get_or_create_global_step()

    # Restore from existing checkpoint
    with tfe.restore_variables_on_create(tf.train.latest_checkpoint(out_dir)):
        gp, hyper_params = util.construct_gp(args, dataset.input_dim, dataset.output_dim, dataset.lik,
                                             dataset.inducing_inputs, dataset.num_train)

    with tf.device(device):
        step = 0
        epoch = 1
        while step < args['train_steps']:
            start = time.time()
            fit(gp, optimizer, dataset, step_counter, hyper_params, args)  # train for one epoch
            end = time.time()
            step = step_counter.numpy()
            if epoch % args['eval_epochs'] == 0 or not step < args['train_steps']:
                print(f"Train time for epoch #{epoch} (global step {step}): {end - start:0.2f}s")
                evaluate(gp, dataset, args)
            if step % args['chkpnt_steps'] == 0 or not step < args['train_steps']:
                all_variables = (gp.get_all_variables() + optimizer.variables() + [step_counter] + hyper_params)
                tfe.Saver(all_variables).save(checkpoint_prefix, global_step=step_counter)
            epoch += 1

        if args['plot'] or args['preds_path']:  # Create predictions
            tf.reset_default_graph()
            mean, var = predict(dataset.xtest, tf.train.latest_checkpoint(out_dir), dataset, args)

    if args['preds_path']:  # save predictions
        working_dir = out_dir if args['save_dir'] else Path(".")
        np.savez_compressed(working_dir / Path(args['preds_path']), pred_mean=mean, pred_var=var)
    if args['plot']:  # plot
        getattr(util.plot, args['plot'])(mean, var, dataset)
    return gp


def fit(gp, optimizer, dataset, step_counter, hyper_params, args):
    """Trains model on `train_data` using `optimizer`.

    Args:
        gp: gaussian process
        optimizer: tensorflow optimizer
        dataset: dataset
        step_counter: variable to keep track of the training step
        hyper_params: hyper params that should be updated during training
        args: additional parameters
    """

    start = time.time()
    for (batch_num, (features, outputs)) in enumerate(tfe.Iterator(dataset.train_fn().batch(args['batch_size']))):
        # Record the operations used to compute the loss given the input, so that the gradient of the loss with
        # respect to the variables can be computed.
        with tfe.GradientTape() as tape:
            obj_func, inf_params = gp.inference(features, outputs, True)
        # Compute gradients
        all_params = inf_params + hyper_params
        if args['loo_steps'] is not None:
            # Alternate loss between NELBO and LOO
            # TODO: allow `nelbo_steps` to be different from `loo_steps`
            if (step_counter.numpy() // args['loo_steps']) % 2 == 0:
                grads_and_params = zip(tape.gradient(obj_func['NELBO'], all_params), all_params)
            else:
                grads_and_params = zip(tape.gradient(obj_func['LOO_VARIATIONAL'], hyper_params), hyper_params)
        else:
            grads_and_params = zip(tape.gradient(obj_func['loss'], all_params), all_params)
        # Apply gradients
        optimizer.apply_gradients(grads_and_params, global_step=step_counter)

        if args['logging_steps'] is not None and batch_num % args['logging_steps'] == 0:
            print(f"Step #{step_counter.numpy()} ({time.time() - start:.4f} sec)\t", end=' ')
            for loss_name, loss_value in obj_func.items():
                print(f"{loss_name}: {loss_value:.2f}", end=' ')
            print("")  # newline
            start = time.time()



def evaluate(gp, dataset, args):
    """Perform an evaluation of `inf_func` on the examples from `dataset`.

    Args:
        gp: gaussian process
        dataset: dataset
        args: additional parameters
    """
    avg_loss = tfe.metrics.Mean('loss')
    metrics = util.init_metrics(dataset.metric, True)

    for (features, outputs) in tfe.Iterator(dataset.test_fn().batch(args['batch_size'])):
        obj_func, _ = gp.inference(features, outputs, False)
        pred_mean, _ = gp.predict(features)
        avg_loss(sum(obj_func.values()))
        util.update_metrics(metrics, features, outputs, pred_mean)
    print(f"Test set: Average loss: {avg_loss.result()}")
    util.record_metrics(metrics)


def predict(test_inputs, saved_model, dataset_info, args):
    """Predict outputs given test inputs.

    This function can be called from a different module and should still work.

    Args:
        test_inputs: ndarray. Points on which we wish to make predictions. Dimensions: num_test * input_dim.
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
    num_inducing = dataset_info.inducing_inputs.shape[0]

    with tfe.restore_variables_on_create(saved_model):
        # Creating the inference object here will restore the variables from the saved model
        gp, _ = util.construct_gp(args, test_inputs.shape[1], dataset_info.output_dim, dataset_info.lik, num_inducing,
                                  dataset_info.num_train)

    test_inputs = np.array_split(test_inputs, num_batches)
    pred_means = [0.0] * num_batches
    pred_vars = [0.0] * num_batches

    for i in range(num_batches):
        pred_means[i], pred_vars[i] = gp.predict({'input': test_inputs[i]})

    return np.concatenate(pred_means, axis=0), np.concatenate(pred_vars, axis=0)

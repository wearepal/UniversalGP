"""Eager training of GP model."""

import time
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from . import inf, util, cov, lik


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
    checkpoint_prefix = out_dir / Path('ckpt')
    step_counter = tf.train.get_or_create_global_step()

    # Restore from existing checkpoint
    with tfe.restore_variables_on_create(tf.train.latest_checkpoint(out_dir)):
        # Gather parameters
        cov_func = [getattr(cov, args['cov'])(dataset.input_dim, args['length_scale'], iso=not args['use_ard'])
                    for _ in range(dataset.output_dim)]
        lik_func = getattr(lik, args['lik'])()
        hyper_params = lik_func.get_params() + sum([k.get_params() for k in cov_func], [])

        gp = getattr(inf, args['inf'])(cov_func, lik_func, dataset.num_train, dataset.inducing_inputs, args)

    with tf.device(device):
        step = 0
        epoch = 1
        while step < args['train_steps']:
            start = time.time()
            fit(gp, optimizer, dataset.train_fn(), step_counter, hyper_params, args)  # train for one epoch
            end = time.time()
            step = step_counter.numpy()
            if epoch % args['eval_epochs'] == 0 or not step < args['train_steps']:
                print(f"Train time for epoch #{epoch} (global step {step}): {end - start:0.2f}s")
                evaluate(gp, dataset.test_fn(), args)
            if step % args['chkpnt_steps'] == 0 or not step < args['train_steps']:
                all_variables = (gp.get_all_variables() + optimizer.variables() + [step_counter] + hyper_params)
                tfe.Saver(all_variables).save(checkpoint_prefix, global_step=step_counter)
            epoch += 1

        if args['save_vars'] and args['save_dir'] is not None:
            var_collection = {var.name: var.numpy() for var in gp.get_all_variables() + hyper_params}
            np.savez_compressed(out_dir / Path("vars"), **var_collection)
        if args['plot'] is not None:
            tf.reset_default_graph()
            # Create predictions
            mean, var = predict(dataset.xtest, tf.train.latest_checkpoint(out_dir), dataset.num_train,
                                dataset.inducing_inputs.shape[-2], dataset.output_dim, args)
            getattr(util.plot, args['plot'])(mean, var, dataset.xtrain, dataset.ytrain, dataset.xtest, dataset.ytest)
    return gp


def fit(gp, optimizer, train_data, step_counter, hyper_params, args):
    """Trains model on `train_data` using `optimizer`.

    Args:
        gp: gaussian process
        optimizer: tensorflow optimizer
        train_data: training dataset (instance of tf.data.Dataset)
        step_counter: variable to keep track of the training step
        hyper_params: hyper params that should be updated during training
        args: additional parameters
    """

    start = time.time()
    for (batch_num, (inputs, outputs)) in enumerate(tfe.Iterator(train_data.batch(args['batch_size']))):
        # Record the operations used to compute the loss given the input, so that the gradient of the loss with
        # respect to the variables can be computed.
        with tfe.GradientTape() as tape:
            obj_func, inf_params = gp.inference(inputs['input'], outputs, True)
            loss = sum(obj_func.values())
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
            grads_and_params = zip(tape.gradient(loss, all_params), all_params)
        # Apply gradients
        optimizer.apply_gradients(grads_and_params, global_step=step_counter)

        if args['logging_steps'] is not None and batch_num % args['logging_steps'] == 0:
            print(f"Step #{step_counter.numpy()} ({time.time() - start:.4f} sec)\t", end=' ')
            for loss_name, loss_value in obj_func.items():
                print('{}: {}'.format(loss_name, loss_value), end=' ')
            print("")
            start = time.time()


def evaluate(gp, test_data, args):
    """Perform an evaluation of `inf_func` on the examples from `dataset`.

    Args:
        gp: gaussian process
        test_data: test dataset (instance of tf.data.Dataset)
        args: additional parameters
    """
    avg_loss = tfe.metrics.Mean('loss')
    if args['metric'] == 'rmse':
        metric = tfe.metrics.Mean('mse')
        update = lambda mse, pred, label: mse((pred - label)**2)
        result = lambda mse: np.sqrt(mse.result())
    elif args['metric'] == 'soft_accuracy':
        metric = tfe.metrics.Accuracy('accuracy')

        def update(accuracy, pred, label):
            accuracy(tf.argmax(pred, axis=1), tf.argmax(label, axis=1))
        result = lambda accuracy: accuracy.result()
    elif args['metric'] == 'logistic_accuracy':
        metric = tfe.metrics.Accuracy('accuracy')

        def update(accuracy, pred, label):
            accuracy(tf.cast(pred > 0.5, tf.int32), tf.cast(label, tf.int32))
        result = lambda accuracy: accuracy.result()

    for (inputs, outputs) in tfe.Iterator(test_data.batch(args['batch_size'])):
        obj_func, _ = gp.inference(inputs['input'], outputs, False)
        pred_mean, _ = gp.predict(inputs['input'])
        avg_loss(sum(obj_func.values()))
        update(metric, pred_mean, outputs)
    print('Test set: Average loss: {}, {}: {}\n'.format(avg_loss.result(), args['metric'], result(metric)))


def predict(test_inputs, saved_model, num_train, num_inducing, output_dim, args):
    """Predict outputs given test inputs.

    This function can be called from a different module and should still work.

    Args:
        test_inputs: ndarray. Points on which we wish to make predictions. Dimensions: num_test * input_dim.
        saved_model: path to saved model
        num_train: the number of training examples
        num_inducing: the number of inducing inputs
        output_dim: number of output dimensions
        args: additional parameters

    Returns:
        ndarray. The predicted mean of the test inputs. Dimensions: num_test * output_dim.
        ndarray. The predicted variance of the test inputs. Dimensions: num_test * output_dim.
    """
    if args['batch_size'] is None:
        num_batches = 1
    else:
        num_batches = util.ceil_divide(test_inputs.shape[0], args['batch_size'])

    with tfe.restore_variables_on_create(saved_model):
        # Creating the inference object here will restore the variables from the saved model
        cov_func = [getattr(cov, args['cov'])(test_inputs.shape[1], iso=not args['use_ard']) for _ in range(output_dim)]
        lik_func = getattr(lik, args['lik'])()
        gp = getattr(inf, args['inf'])(cov_func, lik_func, num_train, num_inducing, args)

    test_inputs = np.array_split(test_inputs, num_batches)
    pred_means = [0.0] * num_batches
    pred_vars = [0.0] * num_batches

    for i in range(num_batches):
        pred_means[i], pred_vars[i] = gp.predict(test_inputs[i])

    return np.concatenate(pred_means, axis=0), np.concatenate(pred_vars, axis=0)

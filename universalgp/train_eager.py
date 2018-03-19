"""Eager training of GP model."""

import time
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from . import inf, util, cov, lik

FLAGS = tf.app.flags.FLAGS


def train_gp(dataset):
    """
    The function is the main Gaussian Process model.
    """

    # Select device
    device = '/gpu:' + FLAGS.gpus
    if FLAGS.gpus is None or tfe.num_gpus() <= 0:
        device = '/cpu:0'
    print('Using device {}'.format(device))

    optimizer = tf.train.RMSPropOptimizer(learning_rate=FLAGS.lr)

    # Set checkpoint path
    if FLAGS.save_dir is not None:
        out_dir = Path(FLAGS.save_dir) / Path(FLAGS.model_name)
        tf.gfile.MakeDirs(str(out_dir))
    else:
        out_dir = Path(mkdtemp())  # Create temporary directory
    checkpoint_prefix = out_dir / Path('ckpt')
    step_counter = tf.train.get_or_create_global_step()

    # Restore from existing checkpoint
    with tfe.restore_variables_on_create(tf.train.latest_checkpoint(out_dir)):
        # Gather parameters
        cov_func = [getattr(cov, FLAGS.cov)(dataset.input_dim, FLAGS.length_scale, iso=not FLAGS.use_ard)
                    for _ in range(dataset.output_dim)]
        lik_func = getattr(lik, FLAGS.lik)()
        hyper_params = lik_func.get_params() + sum([k.get_params() for k in cov_func], [])

        inf_func = getattr(inf, FLAGS.inf)(cov_func, lik_func, dataset.num_train, dataset.inducing_inputs)

    with tf.device(device):
        step = 0
        epoch = 1
        while step < FLAGS.train_steps:
            start = time.time()
            fit(inf_func, optimizer, dataset.train_fn(), step_counter, hyper_params)  # train for one epoch
            end = time.time()
            step = step_counter.numpy()
            if epoch % FLAGS.eval_epochs == 0 or not step < FLAGS.train_steps:
                print(f"Train time for epoch #{epoch} (global step {step}): {end - start:0.2f}s")
                evaluate(inf_func, dataset.test_fn())
            if step % FLAGS.chkpnt_steps == 0 or not step < FLAGS.train_steps:
                all_variables = (inf_func.get_all_variables() + optimizer.variables() + [step_counter] + hyper_params)
                tfe.Saver(all_variables).save(checkpoint_prefix, global_step=step_counter)
            epoch += 1

        if FLAGS.save_vars and FLAGS.save_dir is not None:
            var_collection = {var.name: var.numpy() for var in inf_func.get_all_variables() + hyper_params}
            np.savez_compressed(out_dir / Path("vars"), **var_collection)
        if FLAGS.plot:
            tf.reset_default_graph()
            # Create predictions
            mean, var = predict(dataset.xtest, tf.train.latest_checkpoint(out_dir), dataset.num_train,
                                dataset.inducing_inputs.shape[-2], dataset.output_dim, FLAGS.batch_size)
            util.simple_1d(mean, var, dataset.xtrain, dataset.ytrain, dataset.xtest, dataset.ytest)
    return inf_func


def fit(inf_func, optimizer, train_data, step_counter, hyper_params):
    """Trains model on `dataset` using `optimizer`.

    Args:
        inf_func: inference function
        optimizer: tensorflow optimizer
        train_data: training dataset (instance of tf.data.Dataset)
        step_counter: variable to keep track of the training step
        hyper_params: hyper params that should be updated during training
    """

    start = time.time()
    for (batch_num, (inputs, outputs)) in enumerate(tfe.Iterator(train_data.batch(FLAGS.batch_size))):
        # Record the operations used to compute the loss given the input, so that the gradient of the loss with
        # respect to the variables can be computed.
        with tfe.GradientTape() as tape:
            obj_func, inf_params = inf_func.inference(inputs['input'], outputs, True)
        # Compute gradients
        all_params = inf_params + hyper_params
        if FLAGS.loo_steps is not None:
            # Alternate loss between NELBO and LOO
            # TODO: allow `nelbo_steps` to be different from `loo_steps`
            if (step_counter.numpy() // FLAGS.loo_steps) % 2 == 0:
                grads_and_params = zip(tape.gradient(obj_func['NELBO'], all_params), all_params)
            else:
                grads_and_params = zip(tape.gradient(obj_func['LOO_VARIATIONAL'], hyper_params), hyper_params)
        else:
            grads_and_params = zip(tape.gradient(obj_func['NELBO'], all_params), all_params)
        # Apply gradients
        optimizer.apply_gradients(grads_and_params, global_step=step_counter)

        if FLAGS.logging_steps is not None and batch_num % FLAGS.logging_steps == 0:
            print(f"Step #{step_counter.numpy()} ({time.time() - start:.4f} sec)\t", end=' ')
            for loss_name, loss_value in obj_func.items():
                print('{}: {}'.format(loss_name, loss_value), end=' ')
            print("")
            start = time.time()


def evaluate(inf_func, test_data):
    """Perform an evaluation of `inf_func` on the examples from `dataset`.

    Args:
        inf_func: inference function
        test_data: test dataset (instance of tf.data.Dataset)
    """
    avg_loss = tfe.metrics.Mean('loss')
    if FLAGS.metric == 'rmse':
        metric = tfe.metrics.Mean('mse')
        update = lambda mse, pred, label: mse((pred - label)**2)
        result = lambda mse: np.sqrt(mse.result())
    elif FLAGS.metric == 'accuracy':
        metric = tfe.metrics.Accuracy('accuracy')
        def update(accuracy, pred, label):
            accuracy(tf.argmax(pred, axis=1, output_type=tf.int64), tf.cast(label, tf.int64))
        result = lambda accuracy: accuracy.result()

    for (inputs, outputs) in tfe.Iterator(test_data.batch(FLAGS.batch_size)):
        obj_func, _ = inf_func.inference(inputs['input'], outputs, False)
        pred_mean, _ = inf_func.predict(inputs['input'])
        avg_loss(sum(obj_func.values()))
        update(metric, pred_mean, outputs)
    print('Test set: Average loss: {}, {}: {}\n'.format(avg_loss.result(), FLAGS.metric, result(metric)))


def predict(test_inputs, saved_model, num_train, num_inducing, output_dim, batch_size=None):
    """Predict outputs given test inputs.

    This function can be called from a different module and should still work.

    Args:
        test_inputs: ndarray. Points on which we wish to make predictions. Dimensions: num_test * input_dim.
        saved_model: path to saved model
        num_train: the number of training examples
        num_inducing: the number of inducing inputs
        output_dim: number of output dimensions
        batch_size: int. The size of the batches we make predictions on. If batch_size is None, predict on the
            entire test set at once.

    Returns:
        ndarray. The predicted mean of the test inputs. Dimensions: num_test * output_dim.
        ndarray. The predicted variance of the test inputs. Dimensions: num_test * output_dim.
    """
    if batch_size is None:
        num_batches = 1
    else:
        num_batches = util.ceil_divide(test_inputs.shape[0], batch_size)

    with tfe.restore_variables_on_create(saved_model):
        # Creating the inference object here will restore the variables from the saved model
        cov_func = [getattr(cov, FLAGS.cov)(test_inputs.shape[1], iso=not FLAGS.use_ard) for _ in range(output_dim)]
        lik_func = getattr(lik, FLAGS.lik)()
        inf_func = getattr(inf, FLAGS.inf)(cov_func, lik_func, num_train, num_inducing)

    test_inputs = np.array_split(test_inputs, num_batches)
    pred_means = [0.0] * num_batches
    pred_vars = [0.0] * num_batches

    for i in range(num_batches):
        pred_means[i], pred_vars[i] = inf_func.predict(test_inputs[i])

    return np.concatenate(pred_means, axis=0), np.concatenate(pred_vars, axis=0)

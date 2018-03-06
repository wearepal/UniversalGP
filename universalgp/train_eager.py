"""Eager training of GP model."""

import time
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from . import inf, util, cov, lik

FLAGS = tf.app.flags.FLAGS


def run(dataset):
    """
    The function is the main Gaussian Process model.
    """
    tfe.enable_eager_execution()  # enable Eager Execution (tensors are evaluated immediately, no need for a session)

    # Select device
    device = '/gpu:' + FLAGS.gpus
    if FLAGS.gpus is None or tfe.num_gpus() <= 0:
        device = '/cpu:0'
    print('Using device {}'.format(device))

    # Gather parameters
    cov_func = [getattr(cov, FLAGS.cov)(dataset['input_dim'], FLAGS.length_scale, iso=not FLAGS.use_ard)
                for _ in range(dataset['output_dim'])]
    lik_func = getattr(lik, FLAGS.lik)()
    inf_func = inf.Variational(cov_func, lik_func, FLAGS.diag_post, FLAGS.num_components, FLAGS.num_samples,
                               FLAGS.optimize_inducing, FLAGS.loo)

    # Get hyper parameters
    hyper_params = lik_func.get_params() + sum([k.get_params() for k in cov_func], [])

    optimizer = tf.train.RMSPropOptimizer(learning_rate=FLAGS.lr)

    # Training steps for NELBO and for LOO
    var_steps = 10
    loo_steps = 10

    if FLAGS.save_dir is not None:
        out_dir = Path(FLAGS.save_dir) / Path(FLAGS.model_name)
        tf.gfile.MakeDirs(str(out_dir))
    else:
        out_dir = Path(mkdtemp())  # Create temporary directory
    checkpoint_prefix = out_dir / Path('ckpt')

    with tf.device(device):
        store = tfe.EagerVariableStore()
        with store.as_default():  # This is necessary because the `inf_func` object doesn't store the variables
            step = 0
            epoch = 1
            while step < FLAGS.train_steps:
                with tfe.restore_variables_on_create(tf.train.latest_checkpoint(out_dir)):
                    global_step = tf.train.get_or_create_global_step()
                    start = time.time()
                    train(inf_func, optimizer, dataset, hyper_params)
                    end = time.time()
                    step = global_step.numpy()
                if epoch % FLAGS.eval_epochs == 0 or not step < FLAGS.train_steps:
                    print(f"Train time for epoch #{epoch} (global step {step}): {end - start:0.2f}s")
                    evaluate(inf_func, dataset)
                all_variables = (store.variables() + optimizer.variables() + [global_step] + hyper_params)
                tfe.Saver(all_variables).save(checkpoint_prefix, global_step=global_step)
                epoch += 1

            if FLAGS.save_vars and FLAGS.save_dir is not None:
                var_collection = {var.name: var.numpy() for var in store.variables() + hyper_params}
                np.savez_compressed(out_dir / Path("vars"), **var_collection)
            if FLAGS.plot:
                # Create predictions
                mean, var = predict(inf_func, dataset['xtest'], dataset)
                util.simple_1d(mean, var, dataset['xtrain'], dataset['ytrain'], dataset['xtest'], dataset['ytest'])


def train(inf_func, optimizer, dataset, hyper_params):
    """Trains model on `dataset` using `optimizer`.

    Args:
        inf_func: inference function
        optimizer: tensorflow optimizer
        dataset: training dataset
        hyper_params: hyper params that should be updated during training
    """

    global_step = tf.train.get_or_create_global_step()

    start = time.time()
    for (batch_num, (inputs, outputs)) in enumerate(tfe.Iterator(dataset['train_fn']().batch(FLAGS.batch_size))):
        # Record the operations used to compute the loss given the input, so that the gradient of the loss with
        # respect to the variables can be computed.
        with tfe.GradientTape() as tape:
            obj_func, _, inf_params = inf_func.inference(inputs['input'], outputs, inputs['input'],
                                                         dataset['num_train'], dataset['inducing_inputs'])
            loss = sum(obj_func.values())
        # TODO: Allow alternating losses (just set `grads_and_params` differently depending on `global_step`).
        # Compute gradients
        grads_and_params = zip(tape.gradient(loss, inf_params + hyper_params), inf_params + hyper_params)
        # apply gradients
        optimizer.apply_gradients(grads_and_params, global_step=global_step)

        if FLAGS.logging_steps is not None and batch_num % FLAGS.logging_steps == 0:
            print(f"Step #{global_step.numpy()} ({time.time() - start:.4f} sec)\t", end=' ')
            for loss_name, loss_value in obj_func.items():
                print('{}: {}'.format(loss_name, loss_value), end=' ')
            print("")
            start = time.time()


def evaluate(inf_func, dataset):
    """Perform an evaluation of `inf_func` on the examples from `dataset`.

    Args:
        inf_func: inference function
        dataset: test dataset
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

    for (inputs, outputs) in tfe.Iterator(dataset['test_fn']().batch(FLAGS.batch_size)):
        obj_func, predictions, _ = inf_func.inference(inputs['input'], outputs, inputs['input'],
                                                      dataset['num_train'], dataset['inducing_inputs'])
        avg_loss(sum(obj_func.values()))
        # accuracy(tf.argmax(predictions, axis=1, output_type=tf.int64), tf.cast(outputs, tf.int64))
        update(metric, predictions[0], outputs)
    print('Test set: Average loss: {}, {}: {}\n'.format(avg_loss.result(), FLAGS.metric, result(metric)))


def predict(inf_func, test_inputs, dataset, batch_size=None):
    """Predict outputs given inputs.

    Args:
        inf_func: inference function
        test_inputs: ndarray. Points on which we wish to make predictions. Dimensions: num_test * input_dim.
        dataset: subclass of tf.data.Dataset. The train inputs and outputs.
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

    test_inputs = np.array_split(test_inputs, num_batches)
    pred_means = [0.0] * num_batches
    pred_vars = [0.0] * num_batches

    for (inputs, outputs) in tfe.Iterator(dataset['train_fn']().batch(1).take(1)):
        train_input = inputs['input']
        train_output = outputs
    for i in range(num_batches):
        _, predictions, _ = inf_func.inference(train_input, train_output, test_inputs[i], dataset['num_train'],
                                               dataset['inducing_inputs'])
        pred_means[i], pred_vars[i] = predictions

    return np.concatenate(pred_means, axis=0), np.concatenate(pred_vars, axis=0)

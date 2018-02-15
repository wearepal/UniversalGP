"""
Main training loop for a Gaussian Process
"""
from os import path
import tensorflow as tf
import numpy as np

from universalgp import inf, cov, lik
import datasets

FLAGS = tf.app.flags.FLAGS
### GP flags
tf.app.flags.DEFINE_string('data', 'simple_example', 'Dataset to use')
# tf.app.flags.DEFINE_string('data', 'mnist', 'Dataset to use')
tf.app.flags.DEFINE_string('inf', 'Variational', 'Inference method')
# tf.app.flags.DEFINE_string('inf', 'Exact', 'Inference method')
tf.app.flags.DEFINE_string('lik', 'LikelihoodGaussian', 'Likelihood function')
# tf.app.flags.DEFINE_string('lik', 'LikelihoodSoftmax', 'Likelihood function')
tf.app.flags.DEFINE_string('cov', 'SquaredExponential', 'Covariance function')
tf.app.flags.DEFINE_float('lr', 0.005, 'Learning rate')
tf.app.flags.DEFINE_boolean('use_ard', True, 'Whether to use an automatic relevance determination kernel')
tf.app.flags.DEFINE_float('length_scale', 1.0, 'Initial lenght scale for the kernel')
tf.app.flags.DEFINE_string('metric', 'rmse', 'metric for evaluating the trained model')
### Variational inference flags
tf.app.flags.DEFINE_integer('num_components', 1, 'Number of mixture of Gaussians components')
tf.app.flags.DEFINE_integer('num_samples', 100, 'Number of samples for mean and variance estimate of likelihood')
tf.app.flags.DEFINE_boolean('diag_post', False, 'Whether the Gaussian mixture uses diagonal covariance')
### Tensorflow flags
tf.app.flags.DEFINE_string('model_name', 'local', 'Name of model (used for name of checkpoints)')
tf.app.flags.DEFINE_integer('batch_size', 50, 'Batch size')
tf.app.flags.DEFINE_integer('train_steps', 500, 'Number of training steps')
tf.app.flags.DEFINE_integer('epochs', 10000, 'Number of training epochs')
tf.app.flags.DEFINE_integer('summary_steps', 100, 'How many steps between saving summary')
tf.app.flags.DEFINE_integer('chkpnt_steps', 5000, 'How many steps between saving checkpoints')
tf.app.flags.DEFINE_string('save_dir', None,  # '/its/home/tk324/tensorflow/',
                           'Directory where the checkpoints and summaries are saved')
tf.app.flags.DEFINE_boolean('plot', True, 'Whether to plot the result')
tf.app.flags.DEFINE_integer('logging_steps', 1, 'How many steps between logging the loss')


def build_gaussian_process(features, labels, mode, params: dict):
    """Create the Gaussian Process

    This function is called 3 times to create 3 different graphs which share some variables. It is called with
    mode == TRAIN, mode == EVAL and mode == PREDICT.
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
        inf_func = inf.Variational(cov_func, lik_func, FLAGS.diag_post, FLAGS.num_components, FLAGS.num_samples)
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
    raw_likelihood_params = lik_func.get_params()
    raw_kernel_params = sum([k.get_params() for k in cov_func], [])

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
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step(),
                                  var_list=inf_param + [raw_likelihood_params, raw_kernel_params])
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[
        tf.train.LoggingTensorHook(obj_func, every_n_iter=FLAGS.logging_steps)])


def main():
    """The main entry point

    This functions calls other functions as necessary to construct a graph and then runs the training loop.
    """
    data = getattr(datasets, FLAGS.data)()

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
        model_dir=None if FLAGS.save_dir is None else path.join(FLAGS.save_dir, FLAGS.model_name),
        config=tf.estimator.RunConfig().replace(
            save_checkpoints_secs=None,
            save_checkpoints_steps=FLAGS.chkpnt_steps,
            save_summary_steps=FLAGS.summary_steps,
            keep_checkpoint_max=5,
            log_step_count_steps=FLAGS.chkpnt_steps,
            session_config=tf.ConfigProto(allow_soft_placement=True)))

    # Settings for training
    trainer = tf.estimator.TrainSpec(input_fn=lambda: data['train_fn']().repeat(FLAGS.epochs).batch(FLAGS.batch_size),
                                     max_steps=FLAGS.train_steps)

    # Settings for evaluation
    evaluator = tf.estimator.EvalSpec(input_fn=lambda: data['test_fn']().batch(FLAGS.batch_size))

    tf.estimator.train_and_evaluate(gp, trainer, evaluator)  # this can be replaced by a loop that calls gp.train()

    print(f"length scale = {gp.get_variable_value('cov_se_parameters/length_scale')}")  # print final length scale
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
        from matplotlib import pyplot as plt
        out_dims = len(data['ytrain'][0])
        in_dim = 0
        for i in range(out_dims):
            plt.subplot(out_dims, 1, i + 1)
            plt.plot(data['xtrain'][:, in_dim], data['ytrain'][:, i], '.', mew=2, label='trainings')
            plt.plot(data['xtest'][:, in_dim], data['ytest'][:, i], 'o', mew=2, label='tests')
            plt.plot(data['xtest'][:, in_dim], pred_mean[:, i], 'x', mew=2, label='predictions')

            upper_bound = pred_mean[:, i] + 1.96 * np.sqrt(pred_var[:, i])
            lower_bound = pred_mean[:, i] - 1.96 * np.sqrt(pred_var[:, i])

            plt.fill_between(data['xtest'][:, in_dim], lower_bound, upper_bound, color='gray', alpha=.3, label='95% CI')
        plt.legend(loc='lower left')
        plt.show()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()

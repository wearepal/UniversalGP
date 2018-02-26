import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from universalgp import cov, lik
from universalgp import inf as inference

try:
    tfe.enable_eager_execution()
except ValueError:
    pass


SIG_FIGS = 5


def construct_input():
    train_inputs = tf.constant([[-1], [1]], dtype=tf.float32)
    train_outputs = tf.constant([[1], [-1]], dtype=tf.float32)
    test_inputs = tf.constant([[0]], dtype=tf.float32)
    num_train = tf.constant(2, tf.float32)
    inducing_inputs = np.array([[-1], [1]])
    return [train_inputs, train_outputs, test_inputs, num_train, inducing_inputs]


def test_variational_complete():
    # construct objects
    likelihood = lik.LikelihoodGaussian(1.0)
    kernel = [cov.SquaredExponential(input_dim=1, length_scale=0.5, sf=1.0)]
    vi = inference.Variational(num_samples=100000, lik_func=likelihood, cov_func=kernel, num_components=1,
                               optimize_inducing=False, use_loo=True)

    # compute losses and predictions
    losses, preds, _ = vi.inference(*construct_input())
    nelbo = losses['NELBO']
    loo = losses['LOO_VARIATIONAL']
    pred_mean, pred_var = preds

    # check results
    np.testing.assert_almost_equal(nelbo.numpy(), 4.1, decimal=1)
    np.testing.assert_almost_equal(loo.numpy(), 9.0, decimal=0)
    np.testing.assert_almost_equal(pred_mean.numpy(), 0.0, SIG_FIGS)
    np.testing.assert_almost_equal(tf.squeeze(pred_var).numpy(), 2.0, decimal=3)


def test_exact_complete():
    # construct objects
    likelihood = lik.LikelihoodGaussian(1.0)
    kernel = [cov.SquaredExponential(input_dim=1, length_scale=0.5, sf=1.0)]
    exact = inference.Exact(lik_func=likelihood, cov_func=kernel)

    # compute losses and predictions
    losses, preds, _ = exact.inference(*construct_input())
    nlml = losses['NLML']
    pred_mean, pred_var = preds

    # check results
    np.testing.assert_almost_equal(nlml.numpy(), 2.34046, SIG_FIGS)
    np.testing.assert_almost_equal(pred_mean.numpy(), 0.0, SIG_FIGS)
    np.testing.assert_almost_equal(tf.squeeze(pred_var).numpy(), 1.981779, SIG_FIGS)

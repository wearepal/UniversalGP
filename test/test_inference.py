"""This test does a complete run of the inference function"""

import numpy as np
import tensorflow as tf

from universalgp import cov, lik
from universalgp import inf as inference

try:
    tf.enable_eager_execution()
except ValueError:
    pass


RTOL = 1e-5


def construct_input():
    train_inputs = tf.constant([[-1], [1]], dtype=tf.float32)
    train_outputs = tf.constant([[1], [-1]], dtype=tf.float32)
    test_inputs = tf.constant([[0]], dtype=tf.float32)
    num_train = 2
    inducing_inputs = np.array([[-1], [1]])
    return {'input': train_inputs}, train_outputs, {'input': test_inputs}, num_train, inducing_inputs


def test_variational_complete():
    # construct objects
    train_inputs, train_outputs, test_inputs, num_train, inducing_inputs = construct_input()
    likelihood = lik.LikelihoodGaussian({'sn': 1.0})
    kernel = [cov.SquaredExponential(input_dim=1, args=dict(length_scale=0.5, sf=1.0, iso=False))]
    vi = inference.Variational(kernel, likelihood, num_train, inducing_inputs,
                               {'num_samples': 5000000, 'num_components': 1, 'optimize_inducing': False,
                                'use_loo': True, 'diag_post': False})

    # compute losses and predictions
    losses, _ = vi.inference(train_inputs, train_outputs, True)
    nelbo = losses['NELBO']
    loo = losses['LOO_VARIATIONAL']
    pred_mean, pred_var = vi.predict(test_inputs)

    # check results
    np.testing.assert_allclose(nelbo.numpy(), 3.844, rtol=1e-3)
    np.testing.assert_allclose(loo.numpy(), 4.45, rtol=1e-2)  # test with a relative tolerance of 1%
    np.testing.assert_allclose(pred_mean.numpy(), 0.0, RTOL)
    np.testing.assert_allclose(tf.squeeze(pred_var).numpy(), 2.0, 1e-2)


def test_exact_complete():
    # construct objects
    train_inputs, train_outputs, test_inputs, num_train, _ = construct_input()
    likelihood = lik.LikelihoodGaussian({'sn': 1.0})
    kernel = [cov.SquaredExponential(input_dim=1, args=dict(length_scale=0.5, sf=1.0, iso=False))]
    exact = inference.Exact(kernel, likelihood, num_train)

    # compute losses and predictions
    losses, _ = exact.inference(train_inputs, train_outputs, True)
    nlml = losses['NLML']
    pred_mean, pred_var = exact.predict(test_inputs)

    # check results
    np.testing.assert_allclose(nlml.numpy(), 2.34046, RTOL)
    np.testing.assert_allclose(pred_mean.numpy(), 0.0, RTOL)
    np.testing.assert_allclose(tf.squeeze(pred_var).numpy(), 1.981779, RTOL)

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

RTOL = 1e-5


def chol_normal_log_prob(val, mean, covar):
    chol_normal = tfd.MultivariateNormalTriL(
        tf.constant(mean, dtype=tf.float32), tf.constant(covar, dtype=tf.float32))
    return chol_normal.log_prob(np.array(val, dtype=np.float32)).numpy()


def diag_normal_log_prob(val, mean, covar):
    diag_normal = tfd.MultivariateNormalDiag(
        tf.constant(mean, dtype=tf.float32), tf.sqrt(tf.constant(covar, dtype=tf.float32)))
    return diag_normal.log_prob(np.array(val, dtype=np.float32)).numpy()


class TestCholNormal:
    def test_same_mean(self):
        log_prob = chol_normal_log_prob([1.0], [1.0], [[1.0]])
        np.testing.assert_allclose(log_prob, -0.5 * np.log(2 * np.pi), RTOL)

    def test_scalar_covar(self):
        log_prob = chol_normal_log_prob([1.0], [1.0], [[np.sqrt(2.0)]])
        np.testing.assert_allclose(log_prob, -0.5 * (np.log(2 * np.pi) + np.log(2.0)), RTOL)

    def test_small_scalar_covar(self):
        log_prob = chol_normal_log_prob([1.0], [1.0], [[1e-10]])
        np.testing.assert_allclose(log_prob, -0.5 * (np.log(2 * np.pi) + np.log(1e-20)), RTOL)

    def test_large_scalar_covar(self):
        log_prob = chol_normal_log_prob([1.0], [1.0], [[1e10]])
        np.testing.assert_allclose(log_prob, -0.5 * (np.log(2 * np.pi) + np.log(1e20)), RTOL)

    def test_multi_covar_same_mean(self):
        log_prob = chol_normal_log_prob([1.0, 2.0], [1.0, 2.0], [[1.0, 0.0], [2.0, 3.0]])
        np.testing.assert_allclose(log_prob, -0.5 * (2.0 * np.log(2 * np.pi) + np.log(9.0)), RTOL)


class TestDiagNormal:
    def test_simple(self):
        log_prob = diag_normal_log_prob([-.5, .5], [0., 0.], [4., 4.])
        np.testing.assert_allclose(log_prob, 2 * np.log(0.193334), 1e-5)

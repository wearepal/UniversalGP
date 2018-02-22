import numpy as np
import tensorflow as tf

from universalgp import util
import tensorflow.contrib.eager as tfe

try:
    tfe.enable_eager_execution()
except ValueError:
    pass


SIG_FIGS = 5


def chol_normal_log_prob(val, mean, covar):
    chol_normal = util.CholNormal(tf.constant(mean, dtype=tf.float32), tf.constant(covar, dtype=tf.float32))
    return chol_normal.log_prob(np.array(val, dtype=np.float32)).numpy()


class TestCholNormal:
    def test_same_mean(self):
        log_prob = chol_normal_log_prob([1.0], [1.0], [[1.0]])
        np.testing.assert_almost_equal(log_prob, -0.5 * np.log(2 * np.pi), SIG_FIGS)

    def test_scalar_covar(self):
        log_prob = chol_normal_log_prob([1.0], [1.0], [[np.sqrt(2.0)]])
        np.testing.assert_almost_equal(log_prob, -0.5 * (np.log(2 * np.pi) + np.log(2.0)), SIG_FIGS)

    def test_small_scalar_covar(self):
        log_prob = chol_normal_log_prob([1.0], [1.0], [[1e-10]])
        np.testing.assert_almost_equal(log_prob, -0.5 * (np.log(2 * np.pi) + np.log(1e-20)), SIG_FIGS)

    def test_large_scalar_covar(self):
        log_prob = chol_normal_log_prob([1.0], [1.0], [[1e10]])
        np.testing.assert_almost_equal(log_prob, -0.5 * (np.log(2 * np.pi) + np.log(1e20)), SIG_FIGS)

    def test_multi_covar_same_mean(self):
        log_prob = chol_normal_log_prob([1.0, 2.0], [1.0, 2.0], [[1.0, 0.0], [2.0, 3.0]])
        np.testing.assert_almost_equal(log_prob, -0.5 * (2.0 * np.log(2 * np.pi) + np.log(9.0)), SIG_FIGS)

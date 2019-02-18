"""
Test for computation of entropy
"""

import numpy as np
import scipy.linalg as sl
import scipy.special

import tensorflow as tf

import universalgp


class Normal:
    def __init__(self, mean, covar):
        self.mean = mean
        self.covar = covar


class CholNormal(Normal):
    def prob(self, val):
        return np.exp(self.log_prob(val))

    def log_prob(self, val):
        dim = np.shape(self.mean)[0] + 0.0
        diff = np.expand_dims(val - self.mean, 1)
        chol = self.covar

        quad_form = np.sum(diff * sl.solve_triangular(chol, diff, lower=True))
        log_cholesky_det = 2 * np.sum(np.log(np.diagonal(chol)))

        lp = -0.5 * (dim * np.log(2.0 * np.pi) + log_cholesky_det + quad_form)
        return lp


def mat_square(mat):
    return mat @ np.transpose(mat)


def construct_inf(num_latents, num_components, inducing_inputs):
    """Construct a very basic inference object"""
    return universalgp.inf.Variational(
        dict(num_components=num_components, optimize_inducing=True, num_samples=10,
             diag_post=False, use_loo=False),
        'LikelihoodGaussian', num_latents, 1, inducing_inputs
    )


def tf_constant(np_array):
    return tf.constant(np_array, dtype=tf.float32)


def test_entropy1():
    means = np.array([[[01.0, 02.0],
                       [03.0, 04.0]],
                      [[05.0, 06.0],
                       [07.0, 08.0]]])

    chol_covars = np.array([[[[0.1, 0.0],
                              [0.2, 0.3]],
                             [[0.4, 0.0],
                              [0.5, 0.6]]],
                            [[[0.7, 0.0],
                              [0.8, 0.9]],
                             [[1.0, 0.0],
                              [1.1, 1.2]]]])

    num_components = 2
    num_latents = 2

    log_normal_probs = np.zeros([2, 2])
    for i in range(num_components):
        for j in range(num_components):
            for k in range(num_latents):
                if i == j:
                    # Compute chol(2S) = sqrt(2)*chol(S).
                    chol_covars_sum = np.sqrt(2.0) * chol_covars[i, k, :, :]
                else:
                    covars_sum = (mat_square(chol_covars[i, k, :, :]) +
                                  mat_square(chol_covars[j, k, :, :]))
                    chol_covars_sum = sl.cholesky(covars_sum)

                chol_normal = CholNormal(means[i, k, :], chol_covars_sum)
                log_normal_probs[i][j] += chol_normal.log_prob(means[j, k, :])

    entropy_np = 0.0
    weights = np.array([0.7, 0.3])

    for i in range(num_components):
        weighted_log_probs = np.array([0.0, 0.0])
        for j in range(num_components):
            if i <= j:
                weighted_log_probs[j] = np.log(weights[j]) + log_normal_probs[i][j]
            else:
                weighted_log_probs[j] = np.log(weights[j]) + log_normal_probs[j][i]

        entropy_np -= weights[i] * scipy.special.logsumexp(np.stack(weighted_log_probs))
        # print(entropy)

    inf = construct_inf(num_latents, num_components, 1)
    entropy_tf = inf._build_entropy(tf_constant(weights), tf_constant(means), tf_constant(chol_covars))
    tf.compat.v1.reset_default_graph()

    np.testing.assert_allclose(entropy_tf.numpy(), entropy_np)

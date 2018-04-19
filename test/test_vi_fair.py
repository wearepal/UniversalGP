import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from universalgp import cov, lik
from universalgp import inf as inference

try:
    tfe.enable_eager_execution()
except ValueError:
    pass


RTOL = 1e-4
PARAMS = {'num_components': 1, 'diag_post': False}


def construct_simple_full(biased_acceptance1, biased_acceptance2, target_rate1, target_rate2):
    likelihood = lik.LikelihoodGaussian({'sn': 1.0})
    kernel = [cov.SquaredExponential(input_dim=1, args=dict(length_scale=1.0, sf=1.0, iso=False))]
    # In most of our unit test, we will replace this value with something else.
    return inference.VariationalYbar(kernel, likelihood, 1, 1, {
        **PARAMS, 'target_rate1': target_rate1, 'target_rate2': target_rate2, 'biased_acceptance1': biased_acceptance1,
        'biased_acceptance2': biased_acceptance2, 'probs_from_flipped': False})


def invert(probs):
    probs = np.array(probs)
    return np.stack((1 - probs, probs), 0)


def construct(*, p_y1_ybar0_s0, p_y1_ybar1_s0, p_y1_ybar0_s1, p_y1_ybar1_s1):
    return invert([[p_y1_ybar0_s0, p_y1_ybar1_s0],
                   [p_y1_ybar0_s1, p_y1_ybar1_s1]])


class TestDebiasParams:
    def test_extreme1(self):
        actual = construct_simple_full(0.7, 0.7, 0.7, 0.7)._debiasing_parameters().numpy()
        correct = construct(p_y1_ybar0_s0=0.,
                            p_y1_ybar1_s0=1.,
                            p_y1_ybar0_s1=0.,
                            p_y1_ybar1_s1=1.)
        np.testing.assert_allclose(actual, correct, RTOL)

    def test_extreme2(self):
        actual = construct_simple_full(0.5, 0.5, 1e-5, 1 - 1e-5)._debiasing_parameters().numpy()
        correct = construct(p_y1_ybar0_s0=.5,
                            p_y1_ybar1_s0=1.,
                            p_y1_ybar0_s1=0.,
                            p_y1_ybar1_s1=.5)
        np.testing.assert_allclose(actual, correct, RTOL)

    def test_moderate1(self):
        actual = construct_simple_full(0.3, 0.7, 0.5, 0.5)._debiasing_parameters().numpy()
        correct = construct(p_y1_ybar0_s0=.0,
                            p_y1_ybar1_s0=.3 / .5,
                            p_y1_ybar0_s1=(.7 - .5) / .5,
                            p_y1_ybar1_s1=1.)
        np.testing.assert_allclose(actual, correct, RTOL)

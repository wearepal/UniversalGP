import numpy as np
import tensorflow as tf

from universalgp import cov, lik
from universalgp import inf as inference

try:
    tf.compat.v1.enable_eager_execution()
except ValueError:
    pass


RTOL = 1e-4
PARAMS = dict(num_components=1, diag_post=False, sn=1.0, length_scale=1.0, sf=1.0, iso=False,
              cov='SquaredExponential', optimize_inducing=True,
              # probs_from_flipped=False,
              )


def construct_simple_full(biased_rate1, biased_rate2, target_rate1, target_rate2,
                          p_ybary0_or_ybary1_s0=1.0, p_ybary0_or_ybary1_s1=1.0):
    input_dim = 1
    output_dim = 1
    num_train = 1
    args = {**PARAMS, **dict(target_rate1=target_rate1, target_rate2=target_rate2,
                             biased_acceptance1=biased_rate1, biased_acceptance2=biased_rate2,
                             probs_from_flipped=False, p_ybary0_or_ybary1_s0=p_ybary0_or_ybary1_s0,
                             p_ybary0_or_ybary1_s1=p_ybary0_or_ybary1_s1)}
    inf = inference.VariationalYbar(args, 'LikelihoodGaussian', output_dim, num_train, num_train)
    inf.build((num_train, input_dim))
    return inf


def construct_eq_odds(biased_acceptance1, biased_acceptance2, p_ybary0_s0, p_ybary1_s0, p_ybary0_s1,
                      p_ybary1_s1):
    input_dim = 1
    output_dim = 1
    num_train = 1
    args = {**PARAMS, **dict(p_ybary0_s0=p_ybary0_s0, p_ybary0_s1=p_ybary0_s1,
                             p_ybary1_s0=p_ybary1_s0, p_ybary1_s1=p_ybary1_s1,
                             biased_acceptance1=biased_acceptance1,
                             biased_acceptance2=biased_acceptance2)}
    inf = inference.VariationalYbarEqOdds(args, 'LikelihoodGaussian', output_dim, num_train,
                                          num_train)
    inf.build((num_train, input_dim))
    return inf


def invert(probs):
    probs = np.array(probs)
    return np.stack((1 - probs, probs), 0)


def construct(*, p_y0_ybar0_s0, p_y1_ybar1_s0, p_y0_ybar0_s1, p_y1_ybar1_s1):
    return invert([[1 - p_y0_ybar0_s0, p_y1_ybar1_s0],
                   [1 - p_y0_ybar0_s1, p_y1_ybar1_s1]])


class TestDebiasParams:
    @staticmethod
    def test_extreme1():
        actual = construct_simple_full(0.7, 0.7, 0.7, 0.7)._debiasing_parameters().numpy()
        correct = construct(p_y0_ybar0_s0=1.,
                            p_y1_ybar1_s0=1.,
                            p_y0_ybar0_s1=1.,
                            p_y1_ybar1_s1=1.)
        np.testing.assert_allclose(actual, correct, RTOL)

    @staticmethod
    def test_extreme2():
        actual = construct_simple_full(0.5, 0.5, 1e-5, 1 - 1e-5)._debiasing_parameters().numpy()
        correct = construct(p_y0_ybar0_s0=.5,
                            p_y1_ybar1_s0=1.,
                            p_y0_ybar0_s1=1.,
                            p_y1_ybar1_s1=.5)
        np.testing.assert_allclose(actual, correct, RTOL)

    @staticmethod
    def test_moderate1():
        actual = construct_simple_full(0.3, 0.7, 0.5, 0.5)._debiasing_parameters().numpy()
        correct = construct(p_y0_ybar0_s0=1.,
                            p_y1_ybar1_s0=.3 / .5,
                            p_y0_ybar0_s1=1 - (.7 - .5) / .5,
                            p_y1_ybar1_s1=1.)
        np.testing.assert_allclose(actual, correct, RTOL)

    @staticmethod
    def test_precision_target():
        p_y1_s0 = .3
        p_y1_s1 = .9
        p_ybar1_s0 = .5
        p_ybar1_s1 = .6
        prec_s0 = .7
        prec_s1 = .8
        obj = construct_simple_full(p_y1_s0, p_y1_s1, p_ybar1_s0, p_ybar1_s1, prec_s0, prec_s1)
        actual_lik = inference.inf_vi_ybar.positive_label_likelihood(
            obj.args, [p_y1_s0, p_y1_s1], [p_ybar1_s0, p_ybar1_s1])
        np.testing.assert_allclose(actual_lik,
                                   [[(p_ybar1_s0 - prec_s0 * p_y1_s0) / (1 - p_y1_s0),
                                     1 - .8],
                                    [.7,
                                     (p_ybar1_s1 - (1 - prec_s1) * (1 - p_y1_s1)) / p_y1_s1]],
                                   RTOL)
        actual_full = obj._debiasing_parameters().numpy()
        correct = construct(p_y0_ybar0_s0=1 - (1 - prec_s0) * p_y1_s0 / (1 - p_ybar1_s0),
                            p_y1_ybar1_s0=prec_s0 * p_y1_s0 / p_ybar1_s0,
                            p_y0_ybar0_s1=prec_s1 * (1 - p_y1_s1) / (1 - p_ybar1_s1),
                            p_y1_ybar1_s1=1 - (1 - prec_s1) * (1 - p_y1_s1) / p_ybar1_s1)
        print(actual_full)
        print(correct)
        np.testing.assert_allclose(actual_full, correct, RTOL)


class TestEqOddsParams:
    @staticmethod
    def test_extreme1():
        actual = construct_eq_odds(.3, .7, 1., 1., 1., 1.)._debiasing_parameters().numpy()
        correct = construct(p_y0_ybar0_s0=1.,
                            p_y1_ybar1_s0=1.,
                            p_y0_ybar0_s1=1.,
                            p_y1_ybar1_s1=1.)
        np.testing.assert_allclose(actual, correct, RTOL)

    @staticmethod
    def test_extreme2():
        actual = construct_eq_odds(.25, .75, .5, .5, .0, .0)._debiasing_parameters().numpy()
        correct = construct(p_y0_ybar0_s0=.75,
                            p_y1_ybar1_s0=.25,
                            p_y0_ybar0_s1=0.,
                            p_y1_ybar1_s1=0.)
        np.testing.assert_allclose(actual, correct, RTOL)

    @staticmethod
    def test_moderate1():
        actual = construct_eq_odds(.3, .7, .8, 1., .8, 1.)._debiasing_parameters().numpy()
        correct = construct(p_y0_ybar0_s0=1.,
                            p_y1_ybar1_s0=.3 / (.3 + .2 * .7),
                            p_y0_ybar0_s1=1.,
                            p_y1_ybar1_s1=.7 / (.7 + .2 * .3))
        np.testing.assert_allclose(actual, correct, RTOL)

    @staticmethod
    def test_moderate2():
        actual = construct_eq_odds(.1, .7, .8, .6, .4, .5)._debiasing_parameters().numpy()
        correct = construct(p_y0_ybar0_s0=.8 * (1 - .1) / (.8 * (1 - .1) + (1 - .6) * .1),
                            p_y1_ybar1_s0=.6 * .1 / (.6 * .1 + (1 - .8) * (1 - .1)),
                            p_y0_ybar0_s1=.4 * (1 - .7) / (.4 * (1 - .7) + (1 - .5) * .7),
                            p_y1_ybar1_s1=.5 * .7 / (.5 * .7 + (1 - .4) * (1 - .7)))
        np.testing.assert_allclose(actual, correct, RTOL)

import numpy as np
import tensorflow as tf

from universalgp import cov, lik
from universalgp import inf as inference
from universalgp.inf.inf_vi_ybar import debiasing_params_target_rate, debiasing_params_target_tpr


RTOL = 1e-4
ATOL = 1e-5
PARAMS = dict(num_components=1, diag_post=False, sn=1.0, length_scale=1.0, sf=1.0, iso=False,
              cov='SquaredExponential', optimize_inducing=True,
              # probs_from_flipped=False,
              )

def construct_args_rate(biased_acceptance1, biased_acceptance2, target_rate1, target_rate2,
                        p_ybary0_or_ybary1_s0=1.0, p_ybary0_or_ybary1_s1=1.0):
    return dict(biased_acceptance1=biased_acceptance1, biased_acceptance2=biased_acceptance2,
                target_rate1=target_rate1, target_rate2=target_rate2, probs_from_flipped=False,
                p_ybary0_or_ybary1_s0=p_ybary0_or_ybary1_s0,
                p_ybary0_or_ybary1_s1=p_ybary0_or_ybary1_s1)


def invert(probs):
    probs = np.array(probs)
    return np.stack((1 - probs, probs), 0)


def construct(*, p_y0_ybar0_s0, p_y1_ybar1_s0, p_y0_ybar0_s1, p_y1_ybar1_s1):
    return np.log(invert([[1 - p_y0_ybar0_s0, p_y1_ybar1_s0],
                          [1 - p_y0_ybar0_s1, p_y1_ybar1_s1]]))


class TestDebiasParams:
    @staticmethod
    def test_extreme1():
        args = construct_args_rate(biased_acceptance1=0.7, biased_acceptance2=0.7,
                                   target_rate1=0.7, target_rate2=0.7)
        actual = debiasing_params_target_rate(args).numpy()
        correct = construct(p_y0_ybar0_s0=1.,
                            p_y1_ybar1_s0=1.,
                            p_y0_ybar0_s1=1.,
                            p_y1_ybar1_s1=1.)
        np.testing.assert_allclose(actual, correct, RTOL)

    @staticmethod
    def test_extreme2():
        args = construct_args_rate(biased_acceptance1=0.5, biased_acceptance2=0.5,
                                   target_rate1=1e-5, target_rate2=1 - 1e-5)
        actual = debiasing_params_target_rate(args).numpy()
        correct = construct(p_y0_ybar0_s0=.5,
                            p_y1_ybar1_s0=1.,
                            p_y0_ybar0_s1=1.,
                            p_y1_ybar1_s1=.5)
        np.testing.assert_allclose(actual, correct, RTOL)

    @staticmethod
    def test_moderate1():
        args = construct_args_rate(biased_acceptance1=0.3, biased_acceptance2=0.7,
                                   target_rate1=0.5, target_rate2=0.5)
        actual = debiasing_params_target_rate(args).numpy()
        correct = construct(p_y0_ybar0_s0=1.,
                            p_y1_ybar1_s0=.3 / .5,
                            p_y0_ybar0_s1=1 - (.7 - .5) / .5,
                            p_y1_ybar1_s1=1.)
        np.testing.assert_allclose(actual, correct, RTOL, ATOL)

    @staticmethod
    def test_precision_target():
        p_y1_s0 = .3
        p_y1_s1 = .9
        p_ybar1_s0 = .5
        p_ybar1_s1 = .6
        prec_s0 = .7
        prec_s1 = .8
        args = construct_args_rate(biased_acceptance1=0.3, biased_acceptance2=0.9,
                                   target_rate1=0.5, target_rate2=0.6, p_ybary0_or_ybary1_s0=.7,
                                   p_ybary0_or_ybary1_s1=.8)
        actual_lik = inference.inf_vi_ybar.positive_label_likelihood(
            args, [p_y1_s0, p_y1_s1], [p_ybar1_s0, p_ybar1_s1])
        np.testing.assert_allclose(actual_lik,
                                   [[(p_ybar1_s0 - prec_s0 * p_y1_s0) / (1 - p_y1_s0),
                                     1 - .8],
                                    [.7,
                                     (p_ybar1_s1 - (1 - prec_s1) * (1 - p_y1_s1)) / p_y1_s1]],
                                   RTOL)
        actual_full = debiasing_params_target_rate(args).numpy()
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
        args = dict(biased_acceptance1=.3, biased_acceptance2=.7,
                    p_ybary0_s0=1., p_ybary1_s0=1., p_ybary0_s1=1., p_ybary1_s1=1.)
        actual = debiasing_params_target_tpr(args).numpy()
        correct = construct(p_y0_ybar0_s0=1.,
                            p_y1_ybar1_s0=1.,
                            p_y0_ybar0_s1=1.,
                            p_y1_ybar1_s1=1.)
        np.testing.assert_allclose(actual, correct, RTOL)

    @staticmethod
    def test_extreme2():
        args = dict(biased_acceptance1=.25, biased_acceptance2=.75,
                    p_ybary0_s0=.5, p_ybary1_s0=.5, p_ybary0_s1=.0, p_ybary1_s1=.0)
        actual = debiasing_params_target_tpr(args).numpy()
        correct = construct(p_y0_ybar0_s0=.75,
                            p_y1_ybar1_s0=.25,
                            p_y0_ybar0_s1=0.,
                            p_y1_ybar1_s1=0.)
        np.testing.assert_allclose(actual, correct, RTOL)

    @staticmethod
    def test_moderate1():
        args = dict(biased_acceptance1=.3, biased_acceptance2=.7,
                    p_ybary0_s0=.8, p_ybary1_s0=1., p_ybary0_s1=.8, p_ybary1_s1=1.)
        actual = debiasing_params_target_tpr(args).numpy()
        correct = construct(p_y0_ybar0_s0=1.,
                            p_y1_ybar1_s0=.3 / (.3 + .2 * .7),
                            p_y0_ybar0_s1=1.,
                            p_y1_ybar1_s1=.7 / (.7 + .2 * .3))
        np.testing.assert_allclose(actual, correct, RTOL)

    @staticmethod
    def test_moderate2():
        args = dict(biased_acceptance1=.1, biased_acceptance2=.7,
                    p_ybary0_s0=.8, p_ybary1_s0=.6, p_ybary0_s1=.4, p_ybary1_s1=.5)
        actual = debiasing_params_target_tpr(args).numpy()
        correct = construct(p_y0_ybar0_s0=.8 * (1 - .1) / (.8 * (1 - .1) + (1 - .6) * .1),
                            p_y1_ybar1_s0=.6 * .1 / (.6 * .1 + (1 - .8) * (1 - .1)),
                            p_y0_ybar0_s1=.4 * (1 - .7) / (.4 * (1 - .7) + (1 - .5) * .7),
                            p_y1_ybar1_s1=.5 * .7 / (.5 * .7 + (1 - .4) * (1 - .7)))
        np.testing.assert_allclose(actual, correct, RTOL)

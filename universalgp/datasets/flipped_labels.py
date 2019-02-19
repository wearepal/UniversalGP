"""
Synthetic dataset with flipped labels
"""
import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal

from .definition import Dataset, select_training_and_test, sensitive_statistics, make_dataset

tf.compat.v1.app.flags.DEFINE_float('reject_flip_probability', 0.3, '')
tf.compat.v1.app.flags.DEFINE_float('accept_flip_probability', 0.3, '')
tf.compat.v1.app.flags.DEFINE_boolean('flip_sensitive_attribute', False, '')
tf.compat.v1.app.flags.DEFINE_boolean('test_on_ybar', False, '')
SEED = 123


def flipped_labels(flags):
    """Synthetic data with bias"""
    np.random.seed(SEED)  # set the random seed, which can be reproduced again

    num_all = flags['num_all']
    num_train = flags['num_train']
    num_inducing = flags['num_inducing']
    data = _generate_biased_data(num_all, flags['reject_flip_probability'], flags['accept_flip_probability'])

    (xtrain, ytrain, strain, ybartrain), (xtest, ytest, stest, ybartest) = select_training_and_test(num_train, *data)
    sensitive_statistics(ytrain, strain, ytest, stest)

    if flags['test_on_ybar']:
        ytest = ybartest

    if flags['flip_sensitive_attribute']:
        xtest = np.concatenate((xtest, xtest), 0)
        ytest = np.concatenate((ytest, ytest), 0)
        stest = np.concatenate((stest, 1 - stest), 0)

    if flags['s_as_input']:
        inducing_inputs = np.concatenate((xtrain[::num_train // num_inducing], strain[::num_train // num_inducing]), -1)
        input_dim = 3
    else:
        inducing_inputs = xtrain[::num_train // num_inducing]
        input_dim = 2

    return Dataset(
        train=make_dataset({'input': xtrain.astype(np.float32), 'sensitive': strain.astype(np.float32),
                            'ybar': ybartrain.astype(np.float32)}, ytrain.astype(np.float32)),
        test=make_dataset({'input': xtest.astype(np.float32), 'sensitive': stest.astype(np.float32),
                           'ybar': ybartest.astype(np.float32)}, ytest.astype(np.float32)),
        num_train=num_train,
        input_dim=input_dim,
        inducing_inputs=inducing_inputs,
        output_dim=1,
        lik="LikelihoodLogistic",
        metric=("logistic_accuracy,pred_rate_y1_s0,pred_rate_y1_s1,base_rate_y1_s0,base_rate_y1_s1,"
                "logistic_accuracy_ybar,pred_odds_yybar1_s0,pred_odds_yybar1_s1,base_odds_yybar1_s0,"
                "base_odds_yybar1_s1,pred_odds_yybar0_s0,pred_odds_yybar0_s1,base_odds_yybar0_s0,base_odds_yybar0_s1"),
        xtrain=xtrain,
        ytrain=ytrain,
        xtest=xtest,
        ytest=ytest,
        strain=strain,
        stest=stest
    )


def _gaussian_generator(mean, cov, label, num_examples):
    distribution = multivariate_normal(mean=mean, cov=cov)
    x = distribution.rvs(num_examples)
    y = np.ones(num_examples, dtype=int) * label
    return x, y


def _generate_biased_data(n_all, flip_prob_y1_s0, flip_prob_y0_s1):
    """Generate biased data

    Args:
        n_all: number of all data points
        flip_prob_y1_s0: probability that we flip a label from y=1 to y=0 for s=0
        flip_prob_y0_s1: probability that we flip a label from y=0 to y=1 for s=1
    """
    half_num_examples = n_all // 2
    mu_and_sigma_for_ybar0 = [-2, -2], [[10, 1],
                                        [1, 3]]
    mu_and_sigma_for_ybar1 = [2, 2], [[5, 1],
                                      [1, 5]]
    x_for_ybar0, y_for_ybar0 = _gaussian_generator(*mu_and_sigma_for_ybar0, 0, half_num_examples)  # negative class
    x_for_ybar1, y_for_ybar1 = _gaussian_generator(*mu_and_sigma_for_ybar1, 1, half_num_examples)  # positive class

    # sensitive attributes (or groups) for the datapoints
    s_for_ybar0 = np.random.random_integers(0, 1, half_num_examples)  # for y=0
    s_for_ybar1 = np.random.random_integers(0, 1, half_num_examples)  # for y=1

    ybar = np.concatenate((y_for_ybar0, y_for_ybar1))[:, np.newaxis]  # we haven't flipped any labels yet

    # flip the label for some in the positive label for group 0
    y_for_ybar1[np.logical_and(s_for_ybar1 == 0, np.random.random_sample(half_num_examples) < flip_prob_y1_s0)] = 0
    # flip the label for some in the negative label for group 1
    y_for_ybar0[np.logical_and(s_for_ybar0 == 1, np.random.random_sample(half_num_examples) < flip_prob_y0_s1)] = 1

    # join the positive and negative class clusters
    inputs = np.concatenate((x_for_ybar0, x_for_ybar1), axis=0)
    outputs = np.concatenate((y_for_ybar0, y_for_ybar1))[:, np.newaxis]
    sensi_attr = np.concatenate((s_for_ybar0, s_for_ybar1))[:, np.newaxis]

    return inputs, outputs, sensi_attr, ybar

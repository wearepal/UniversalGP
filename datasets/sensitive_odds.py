"""
Usage: Generate the simple synthetic data with two non-sensitive features and one sensitive feature.
       A sensitive feature, 0.0 : protected group (e.g., female)
                            1.0 : non-protected group (e.g., male).
       For equality odds fairness
"""
import numpy as np
from scipy.stats import multivariate_normal

from .definition import Dataset, to_tf_dataset_fn, select_training_and_test

SEED = 2345657


def sensitive_odds_example(_):
    """Simple equality odds example with synthetic data."""
    np.random.seed(SEED)  # set the random seed, which can be reproduced again

    n_all = 250
    inputs, outputs, sensi_attr = _generate_feature(n_all)

    num_train = 200
    (xtrain, ytrain, sensi_attr_train), (xtest, ytest, sensi_attr_test) = select_training_and_test(
        4 * num_train, inputs, outputs[..., np.newaxis], sensi_attr)
    num_inducing = 200

    return Dataset(
        train_fn=to_tf_dataset_fn(xtrain, ytrain, sensi_attr_train),
        test_fn=to_tf_dataset_fn(xtest, ytest, sensi_attr_test),
        num_train=4 * num_train,
        input_dim=2,
        inducing_inputs=xtrain[::num_train // num_inducing],
        output_dim=1,
        lik="LikelihoodLogistic",
        metric="logistic_accuracy",
        xtrain=xtrain,
        ytrain=ytrain,
        xtest=xtest,
        ytest=ytest,
        strain=sensi_attr_train,
        stest=sensi_attr_test
    )


def _gaussian_diff_generator(mean, cov, z_val, label, n):
    distribution = multivariate_normal(mean=mean, cov=cov)
    X = distribution.rvs(n)
    y = np.ones(n, dtype=float) * label
    z = np.ones(n, dtype=float) * z_val  # all the points in this cluster get this value of the
    # sensitive attribute
    return distribution, X, y, z


def _generate_feature(n, data_type=False):
    if data_type:
        """ Generate data: different false possitive rate, but the same false negative rate """
        cov = [[3, 1], [1, 3]]
        mu1, sigma1 = [2, 2], cov  # z=1, +
        mu2, sigma2 = [2, 2], cov  # z=0, +

        mu3, sigma3 = [-2, -2], cov # z=1, -
        cov = [[3, 3], [1, 3]]
        mu4, sigma4 = [-1, 0], cov # z=0, -

    else:
        """ Generate data: different false possitive rate, and different false negative rate """
        cov = [[3, 1], [1, 4]]
        mu1, sigma1 = [2, 3], cov  # z=1, +
        cov = [[5, 2], [2, 5]]
        mu2, sigma2 = [1, 2], cov  # z=0, +

        cov = [[5, 1], [1, 5]]
        mu3, sigma3 = [-5, -1], cov  # z=1, -
        cov = [[3, 1], [1, 3]]
        mu4, sigma4 = [-1, -1], cov  # z=0, -

    nv1, X1, y1, z1 = _gaussian_diff_generator(mu1, sigma1, 1, 1, int(n * 1))  # z=1, +
    nv2, X2, y2, z2 = _gaussian_diff_generator(mu2, sigma2, 0, 1, int(n * 1))  # z=0, +
    nv3, X3, y3, z3 = _gaussian_diff_generator(mu3, sigma3, 1, 0, int(n * 1))  # z=1, -
    nv4, X4, y4, z4 = _gaussian_diff_generator(mu4, sigma4, 0, 0, int(n * 1))  # z=0, -

    # merge the class clusters
    inputs = np.vstack((X1, X2, X3, X4))
    outputs = np.hstack((y1, y2, y3, y4))
    sensi_attr = np.hstack((z1, z2, z3, z4))

    return inputs, outputs, sensi_attr

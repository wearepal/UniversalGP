"""
Generate the simple synthetic data with two non-sensitive features and one sensitive feature.
A sensitive feature, 0.0 : protected group (e.g., female)
                     1.0 : non-protected group (e.g., male).
For parity demographic
"""
import numpy as np
from scipy.stats import multivariate_normal

from .definition import Dataset, select_training_and_test, to_tf_dataset, sensitive_statistics

SEED = 1234


def sensitive_example(_):
    """Synthetic data with bias"""
    np.random.seed(SEED)  # set the random seed, which can be reproduced again

    n_all = 500
    disc_factor = np.pi / 5.0  # discrimination in the data -- decrease it to generate more discrimination
    inputs, outputs, sensi_attr = _generate_feature(n_all, disc_factor)

    num_train = 300
    (xtrain, ytrain, sensi_attr_train), (xtest, ytest, sensi_attr_test) = select_training_and_test(
        2 * num_train, inputs, outputs[..., np.newaxis], sensi_attr[..., np.newaxis])
    num_inducing = 200

    sensitive_statistics(ytrain, sensi_attr_train, ytest, sensi_attr_test)

    return Dataset(
        train_fn=to_tf_dataset(xtrain, ytrain, sensi_attr_train),
        test_fn=to_tf_dataset(xtest, ytest, sensi_attr_test),
        num_train=2 * num_train,
        input_dim=3,
        inducing_inputs=np.concatenate((xtrain[::num_train // num_inducing],
                                        sensi_attr_train[::num_train // num_inducing]), -1),
        output_dim=1,
        lik="LikelihoodLogistic",
        metric=["logistic_accuracy", "pred_rate_y1_s0", "pred_rate_y1_s1", "base_rate_y1_s0", "base_rate_y1_s1",
                "pred_odds_yhaty0_s0", "pred_odds_yhaty0_s1", "pred_odds_yhaty1_s0", "pred_odds_yhaty1_s1"],
        xtrain=xtrain,
        ytrain=ytrain,
        xtest=xtest,
        ytest=ytest,
        strain=sensi_attr_train,
        stest=sensi_attr_test
    )


def _gaussian_generator(mean, cov, label, n):
    distribution = multivariate_normal(mean=mean, cov=cov)
    X = distribution.rvs(n)
    y = np.ones(n, dtype=float) * label
    return distribution, X, y


def _generate_feature(n, disc_factor):
    """Generate the non-sensitive features randomly"""
    mu1, sigma1 = [2, 2], [[5, 1], [1, 5]]
    mu2, sigma2 = [-2, -2], [[10, 1], [1, 3]]
    nv1, X1, y1 = _gaussian_generator(mu1, sigma1, 1, n)  # positive class
    nv2, X2, y2 = _gaussian_generator(mu2, sigma2, 0, n)  # negative class

    # join the positive and negative class clusters
    inputs = np.vstack((X1, X2))
    outputs = np.hstack((y1, y2))

    rotation = np.array([[np.cos(disc_factor), -np.sin(disc_factor)],
                         [np.sin(disc_factor), np.cos(disc_factor)]])
    inputs_aux = inputs @ rotation

    #### Generate the sensitive feature here ####
    sensi_attr = []  # this array holds the sensitive feature value
    for i in range(len(inputs)):
        x = inputs_aux[i]

        # probability for each cluster that the point belongs to it
        p1 = nv1.pdf(x)
        p2 = nv2.pdf(x)

        # normalize the probabilities from 0 to 1
        s = p1 + p2
        p1 = p1 / s
        p2 = p2 / s

        r = np.random.uniform()  # generate a random number from 0 to 1

        if r < p1:  # the first cluster is the positive class
            sensi_attr.append(1.0)  # 1.0 means its male
        else:
            sensi_attr.append(0.0)  # 0.0 -> female

    sensi_attr = np.array(sensi_attr)

    return inputs, outputs, sensi_attr

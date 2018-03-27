import numpy as np
from numpy import random

from .definition import Dataset, select_training_and_test, to_tf_dataset_fn


def sensitive_zhang():
    """
    Toy dataset from Zhang (2017) 'Mitigating Unwanted Biases with Adversarial Learning'
    """
    # parameters
    n_all = 1000
    num_train = 750
    num_inducing = 250
    latent_std = 2.  # standard dev of the latent variable that we're trying to predict. bigger -> easier to predict
    features_std = .5  # standard dev of the features with respect to latent var. smaller -> easier to predict
    raw_output_std = .5  # standard dev of the raw output with respect to latent var. smaller -> easier to predict

    # construction
    sensitive_attr = random.random_integers(0, 1, n_all)
    latent = random.normal(sensitive_attr, latent_std)
    features = random.normal(latent, features_std)
    raw_output = random.normal(latent, raw_output_std)
    labels = (raw_output > 0).astype(np.int)

    xtrain, ytrain, xtest, ytest, strain, stest = select_training_and_test(
        num_train, features[..., np.newaxis], labels[..., np.newaxis], sensitive_attr[..., np.newaxis])

    return Dataset(
        train_fn=to_tf_dataset_fn(xtrain, ytrain, strain),
        test_fn=to_tf_dataset_fn(xtest, ytest, stest),
        num_train=num_train,
        input_dim=1,
        inducing_inputs=xtrain[::num_train // num_inducing],
        output_dim=1,
        lik="LikelihoodLogistic",
        metric="logistic_accuracy",
        xtrain=xtrain,
        ytrain=ytrain,
        xtest=xtest,
        ytest=ytest,
        strain=strain,
        stest=stest
    )

"""Toy dataset with a sensitive attribute"""
import numpy as np
from numpy import random

from .definition import Dataset, select_training_and_test, to_tf_dataset_fn, sensitive_statistics

SEED = 123


def sensitive_zhang(flags):
    """
    Toy dataset from Zhang (2017) 'Mitigating Unwanted Biases with Adversarial Learning'
    """
    # parameters
    n_all = 3000
    num_train = 2000
    num_inducing = flags['num_inducing']  # 500
    # standard dev of the latent variable that we're trying to predict. bigger -> easier to predict
    latent_std = 1.
    # standard dev of the features with respect to latent var. smaller -> easier to predict
    features_std = 1.
    # standard dev of the raw output with respect to latent var. smaller -> easier to predict
    raw_output_std = 1.

    # construction
    np.random.seed(SEED)  # make it reproducible
    sensitive_attr = random.random_integers(0, 1, n_all)
    latent = random.normal(sensitive_attr, latent_std)
    features = random.normal(latent, features_std)
    raw_output = random.normal(latent, raw_output_std)
    labels = (raw_output > 0).astype(np.int)

    (xtrain, ytrain, strain), (xtest, ytest, stest) = select_training_and_test(
        num_train, features[..., np.newaxis], labels[..., np.newaxis],
        sensitive_attr[..., np.newaxis])

    sensitive_statistics(ytrain, strain, ytest, stest)

    return Dataset(
        train_fn=to_tf_dataset_fn(xtrain, ytrain, strain),
        test_fn=to_tf_dataset_fn(xtest, ytest, stest),
        num_train=num_train,
        input_dim=2,
        inducing_inputs=np.concatenate((xtrain[::num_train // num_inducing],
                                        strain[::num_train // num_inducing]), -1),
        output_dim=1,
        lik="LikelihoodLogistic",
        metric="logistic_accuracy,pred_rate_y1_s0,pred_rate_y1_s1,base_rate_y1_s0,base_rate_y1_s1",
        xtrain=xtrain,
        ytrain=ytrain,
        xtest=xtest,
        ytest=ytest,
        strain=strain,
        stest=stest
    )


def sensitive_zhang_simple(flags):
    """
    Simplifaction of the Zhang dataset
    """
    # parameters
    n_all = 3000
    num_train = 2000
    num_inducing = 500
    # standard dev of the latent variable that we're trying to predict. bigger -> easier to predict
    latent_std = 1.
    # standard dev of the features with respect to latent var. smaller -> easier to predict
    features_std = 1.
    sensitive_effect = .5
    # effect of the sensitive attribute on the label. smaller -> easier to predict
    # standard dev of the raw output with respect to latent var. smaller -> easier to predict
    raw_output_std = 1.

    # construction
    random.seed(SEED)  # make it reproducible
    sensitive_attr = random.random_integers(0, 1, n_all)
    latent = random.normal(0, latent_std, n_all)
    features = random.normal(latent, features_std)
    raw_output = random.normal(latent + sensitive_effect * sensitive_attr, raw_output_std)
    labels = (raw_output > 0).astype(np.int)

    (xtrain, ytrain, strain), (xtest, ytest, stest) = select_training_and_test(
        num_train, features[..., np.newaxis], labels[..., np.newaxis],
        sensitive_attr[..., np.newaxis])

    sensitive_statistics(ytrain, strain, ytest, stest)

    return Dataset(
        train_fn=to_tf_dataset_fn(xtrain, ytrain, strain),
        test_fn=to_tf_dataset_fn(xtest, ytest, stest),
        num_train=num_train,
        input_dim=2,
        inducing_inputs=np.concatenate((xtrain[::num_train // num_inducing],
                                        strain[::num_train // num_inducing]), -1),
        output_dim=1,
        lik="LikelihoodLogistic",
        metric="logistic_accuracy,pred_rate_y1_s0,pred_rate_y1_s1,base_rate_y1_s0,base_rate_y1_s1",
        xtrain=xtrain,
        ytrain=ytrain,
        xtest=xtest,
        ytest=ytest,
        strain=strain,
        stest=stest
    )

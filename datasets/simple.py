"""
Simple datasets for testing
"""

import numpy as np

from .definition import Dataset, select_training_and_test, to_tf_dataset_fn, default_feat_col


def simple_example():
    """Simple 1D example with synthetic data."""
    n_all = 200
    num_train = 50
    inputs = np.linspace(0, 5, num=n_all)[:, np.newaxis]
    outputs = np.cos(inputs)
    xtrain, ytrain, xtest, ytest = select_training_and_test(num_train, inputs, outputs)
    num_inducing = 50

    return Dataset(train_fn=to_tf_dataset_fn(xtrain, ytrain),
                   test_fn=to_tf_dataset_fn(xtest, ytest),
                   num_train=num_train,
                   input_dim=1,
                   inducing_inputs=xtrain[::num_train // num_inducing],
                   output_dim=1,
                   lik="LikelihoodGaussian",
                   metric="rmse",
                   **default_feat_col(input_dim=1),
                   xtrain=xtrain,
                   ytrain=ytrain,
                   xtest=xtest,
                   ytest=ytest)


def simple_multi_out():
    """Example with multi-dimensional output."""
    n_all = 200
    num_train = 50
    inputs = np.linspace(0, 5, num=n_all)[:, np.newaxis]
    output1 = np.cos(inputs)
    output2 = np.sin(inputs)
    outputs = np.concatenate((output1, output2), axis=1)
    num_inducing = 50

    xtrain, ytrain, xtest, ytest = select_training_and_test(num_train, inputs, outputs)

    return Dataset(train_fn=to_tf_dataset_fn(xtrain, ytrain),
                   test_fn=to_tf_dataset_fn(xtest, ytest),
                   num_train=num_train,
                   input_dim=1,
                   inducing_inputs=xtrain[::num_train // num_inducing],
                   output_dim=2,
                   lik="LikelihoodGaussian",
                   metric="rmse",
                   **default_feat_col(input_dim=1),
                   xtrain=xtrain,
                   ytrain=ytrain,
                   xtest=xtest,
                   ytest=ytest)


def simple_multi_in():
    """Example with multi-dimensional input."""
    n_all = 200
    num_train = 50
    input1 = np.linspace(-1, 1, num=n_all)[:, np.newaxis]
    input2 = np.linspace(-1, 1, num=n_all)[:, np.newaxis]
    inputs = np.concatenate((input1, input2), 1)
    # outputs = np.cos(input1 + input2)
    outputs = input1**2 + input2**2
    num_inducing = 50

    xtrain, ytrain, xtest, ytest = select_training_and_test(num_train, inputs, outputs)

    return Dataset(train_fn=to_tf_dataset_fn(xtrain, ytrain),
                   test_fn=to_tf_dataset_fn(xtest, ytest),
                   num_train=num_train,
                   input_dim=2,
                   inducing_inputs=xtrain[::num_train // num_inducing],
                   output_dim=1,
                   lik="LikelihoodGaussian",
                   metric="rmse",
                   **default_feat_col(input_dim=2),
                   xtrain=xtrain,
                   ytrain=ytrain,
                   xtest=xtest,
                   ytest=ytest)

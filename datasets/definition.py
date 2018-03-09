"""
This file defines what a dataset should look like.
"""

from typing import NamedTuple, Callable
from numpy import ndarray

class Dataset(NamedTuple):
    """
    Definition of a dataset

    The dataset only has variables and no functions.
    """
    train_fn: Callable  # function that returns training data
    test_fn: Callable  # function that returns the test data
    num_train: int  # number of training instances
    inducing_inputs: ndarray  # initial values for the inducing inputs
    input_dim: int  # number of input dimensions
    output_dim: int  # number of output dimensions
    xtrain: ndarray = None  # (optional) the training input as numpy array
    ytrain: ndarray = None  # (optional) the training output as numpy array
    xtest: ndarray = None  # (optional) the test input as numpy array
    ytest: ndarray = None  # (optional) the test output as numpy array

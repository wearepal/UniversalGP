"""
This file defines what a dataset should look like.
"""

from typing import NamedTuple, Callable, Dict
import numpy as np
import tensorflow as tf

class Dataset(NamedTuple):
    """
    Definition of a dataset

    The dataset only has variables and no functions.
    """
    train_fn: Callable  # function that returns training data
    test_fn: Callable  # function that returns the test data
    num_train: int  # number of training instances
    inducing_inputs: np.ndarray  # initial values for the inducing inputs
    input_dim: int  # number of input dimensions
    output_dim: int  # number of output dimensions
    lik: str  # name of likelihood function
    metric: str # name of the metric to use for evaluation during training
    train_feature_columns: list = None  # a list of feature columns that describe the input during training time
    test_feature_columns: list = None  # a list of feature columns that describe the input during test time
    xtrain: np.ndarray = None  # (optional) the training input as numpy array
    ytrain: np.ndarray = None  # (optional) the training output as numpy array
    xtest: np.ndarray = None  # (optional) the test input as numpy array
    ytest: np.ndarray = None  # (optional) the test output as numpy array
    stest: np.ndarray = None  # sensitive attribute for test
    strain: np.ndarray = None  # sensitive attribute for train


def select_training_and_test(num_train, *data_parts):
    """Randomly devide a dataset into training and test
    Args:
        num_train: Desired number of examples in training set
        *data_parts: Parts of the dataset. The * means that the function can take an arbitrary number of arguments.
    Returns:
        Two lists: data_parts_train, data_parts_test
    """
    idx = np.arange(data_parts[0].shape[0])
    np.random.shuffle(idx)
    train_idx = idx[:num_train]
    test_idx = np.sort(idx[num_train:])

    data_parts_train = []
    data_parts_test = []
    for data_part in data_parts:  # data_parts is a list of the arguments passed to the function
        data_parts_train.append(data_part[train_idx])
        data_parts_test.append(data_part[test_idx])

    return data_parts_train, data_parts_test


def to_tf_dataset_fn(inputs: np.ndarray, outputs: np.ndarray, sensitive=None, dtype_in=np.float32, dtype_out=np.float32,
                     dtype_sen=np.float32):
    """Create a dataset function out of input and output numpy arrays

    It is necessary to wrap the tensorflow code into a function because we have to make sure it's only executed when
    the session has been started. If we just create the dataset here without the `dataset_function` then this will
    produce an (inscrutable) error in the training loop.

    Args:
        inputs: the features as a numpy array
        outputs: the labels as a numpy array
        sensitive: (optional) the sensitive attributes as a numpy array
        dtype_in: (optional) the desired type of the input tensor
        dtype_out: (optional) the desired type of the output tensor
        dtype_sen: (optional) the desired type of the sensitive attribute tensor
    Returns:
        a function that returns the Tensorflow dataset
    """
    inputs_dict = {'input': inputs.astype(dtype_in)}  # the inputs are in a dict so you can add more
    if sensitive is not None:
        inputs_dict.update({'sensitive': sensitive.astype(dtype_sen)})  # add sensitive to input
    return wrap_in_function(inputs_dict, outputs.astype(dtype_out))


def wrap_in_function(inputs_dict: Dict[str, np.ndarray], outputs: np.ndarray) -> Callable:
    """Wrap the given values in a function that creates a Tensorflow dataset from them

    Args:
        input_dict: a dictionary with numpy arrays that must already have the correct type
        output: a numpy array that must already have the correct type
    Returns:
        a function that returns the Tensorflow dataset
    """
    def dataset_function():
        """This function will be called by the training loop"""
        typed_inputs_dict = {input_name: tf.constant(input_value) for input_name, input_value in inputs_dict.items()}
        return tf.data.Dataset.from_tensor_slices((typed_inputs_dict, tf.constant(outputs)))

    return dataset_function


def sensitive_statistics(ytrain, strain, ytest, stest):
    """Compute and print simple statistics about the bias in the dataset
    Args:
        ytrain: labels for training set
        strain: sensitive attritube for training set
        ytest: labels for test set
        stest: sensitive attritube for test set
    """
    for mode, y, s in [("train", ytrain, strain), ("test", ytest, stest)]:
        rate_y1_s0 = np.sum(y[s == 0] == 1) / np.sum(s == 0)
        rate_y1_s1 = np.sum(y[s == 1] == 1) / np.sum(s == 1)
        print(f"{mode} set: P(y=1|s=0) = {rate_y1_s0 * 100:.2f}%")
        print(f"{mode} set: P(y=1|s=1) = {rate_y1_s1 * 100:.2f}%")

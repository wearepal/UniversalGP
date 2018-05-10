"""Dataset with sensitive attribute from numpy files"""
from pathlib import Path

import numpy as np
import tensorflow as tf

from .definition import Dataset, to_tf_dataset_fn, DATA

tf.app.flags.DEFINE_string('dataset_dir', '', 'Directory where the the data is')

MAX_NUM_INDUCING = 500  # maximum number of inducing inputs


def sensitive_from_numpy(flags):
    """Load all data from `dataset_dir` and then construct a dataset

    You must specify a path to a directory in the flag `dataset_dir`. In this directory there must be the file
    'data.npz'. This file must be a numpy file with the following numpy arrays: 'xtrain', 'ytrain', 'strain', 'xtest',
    'ytest', 'stest'.
    """
    data_path = Path(flags['dataset_dir'])

    # Load data from `<dataset_dir>/data.npz`
    raw_data = np.load(data_path / Path("data.npz"))

    # Normalize input and create DATA tuples for easier handling
    input_normalizer = _get_normalizer(raw_data['xtrain'])
    train = DATA(x=input_normalizer(raw_data['xtrain']), y=raw_data['ytrain'], s=raw_data['strain'])
    test = DATA(x=input_normalizer(raw_data['xtest']), y=raw_data['ytest'], s=raw_data['stest'])

    # The following is a bit complicated and could be improved. First, we construct the inducing inputs from the
    # separated data. Then, we call `_merge_x_and_s` depending on what kind of GP we have. The problem here is that
    # sometimes we want the inducing inputs to have merged input but not the training data.
    inducing_inputs = _inducing_inputs(train, flags.get('s_as_input', False))
    if flags['inf'] not in ['VariationalYbar', 'VariationalYbarEqOdds'] and flags.get('s_as_input', False):
        train, test = [_merge_x_and_s(prepared_data) for prepared_data in [train, test]]

    return Dataset(
        train_fn=to_tf_dataset_fn(train.x, train.y, train.s),
        test_fn=to_tf_dataset_fn(test.x, test.y, test.s),
        input_dim=inducing_inputs.shape[1],
        # xtrain=train.x,
        # ytrain=train.y,
        # strain=train.s,
        xtest=test.x,  # needed for making predictions
        # ytest=test.y,
        # stest=test.s,
        num_train=train.x.shape[0],
        inducing_inputs=inducing_inputs,
        output_dim=train.y.shape[1],
        lik="LikelihoodLogistic",
        metric="logistic_accuracy,pred_rate_y1_s0,pred_rate_y1_s1,base_rate_y1_s0,base_rate_y1_s1",
    )


def _merge_x_and_s(data):
    """Merge the input and the sensitive attributes"""
    merged_input = np.concatenate((data.x, data.s), -1)
    return DATA(x=merged_input, y=data.y, s=data.s)


def _inducing_inputs(train, s_as_input):
    """Construct inducing inputs

    This could be done more cleverly with k means

    Args:
        train: the training data
        s_as_input: whether or not the sensitive attribute is part of the input

    Returns:
        inducing inputs
    """
    num_train = train.x.shape[0]
    num_inducing = min(num_train, MAX_NUM_INDUCING)
    if s_as_input:
        return np.concatenate((train.x[::num_train // num_inducing], train.s[::num_train // num_inducing]), -1)
    return train.x[::num_train // num_inducing]


def _get_normalizer(base):
    """Construct normalizer to prevent Cholesky problems"""
    if base.min() == 0 and base.max() > 10:
        max_per_feature = np.amax(base, axis=0)

        def normalizer(unnormalized):
            return np.where(max_per_feature > 1e-7, unnormalized / max_per_feature, unnormalized)
        return normalizer

    def do_nothing(inp):
        return inp
    return do_nothing

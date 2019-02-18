"""Functions for helping with training"""
from pathlib import Path
import numpy as np
import tensorflow as tf

from . import plot
from .. import cov, inf, lik


def construct_from_flags(flags, dataset, inducing_inputs):
    """Construct the necessary objects from the information in the flags

    Args:
        flags: dictionary with parameters
        dataset: information about the data
        inducing_inputs: inducing inputs
    Returns:
        a GP object
    """
    return getattr(inf, flags['inf'])(flags, dataset.lik, dataset.output_dim, dataset.num_train,
                                      inducing_inputs)


def construct_lik_and_cov(gp_obj, flags, lik_name, input_dim, output_dim):
    """Construct the likelihood and all covariance functions from the given information

    Args:
        gp_obj: the GP object that the constructed functions will belong to
        flags: dictionary with parameters
        lik_name: name of the likelihood function
        input_dim: number of input dimensions
        output_dim: number of output dimensions
    Returns:
        a likelihood function and a list of covariance functions
    """
    cov_func = [getattr(cov, flags['cov'])(gp_obj, input_dim, flags) for _ in range(output_dim)]
    lik_func = getattr(lik, lik_name)(gp_obj, flags)
    return lik_func, cov_func


def get_optimizer(flags, global_step):
    """Construct the optimizer from the information in the flags

    Args:
        flags: dictionary with parameters
        global_step: the step in training
    Returns:
        the optimizer and a function to update the learning rate
    """
    schedule = [flags['lr'], flags['lr'] * flags['lr_drop_factor']]
    drop_steps = flags['lr_drop_steps']

    if drop_steps > 0:
        learning_rate = tf.compat.v1.train.piecewise_constant(global_step, [drop_steps], schedule)
    else:
        learning_rate = flags['lr']
    return getattr(tf.train, flags['optimizer'])(learning_rate)


def post_training(pred_mean, pred_var, out_dir, dataset, flags):
    """Call all functions that need to be executed after training has finished

    Args:
        pred_mean: predicted mean
        pred_var: predicted variance
        out_dir: path where to store predictions or None
        dataset: dataset object
        flags: dictionary with parameters
    """
    working_dir = Path(out_dir) if flags['save_dir'] else Path(".")
    with open(working_dir / Path(f"flag_{flags['model_name']}.txt"), 'w') as f:
        flagstr = [f"--{k}={v}" for k, v in flags.items() if not (k.startswith("help") or k == "h")]
        f.write("\n".join(flagstr))
    if flags['preds_path']:
        np.savez_compressed(working_dir / Path(flags['preds_path']),
                            pred_mean=pred_mean, pred_var=pred_var)
    if flags['plot']:
        getattr(plot, flags['plot'])(pred_mean, pred_var, dataset)

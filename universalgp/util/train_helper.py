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
        a GP object and the hyper parameters
    """
    cov_func = [getattr(cov, flags['cov'])(dataset.input_dim, flags)
                for _ in range(dataset.output_dim)]
    lik_func = getattr(lik, dataset.lik)(flags)
    hyper_params = lik_func.get_params() + sum([k.get_params() for k in cov_func], [])

    gp = getattr(inf, flags['inf'])(cov_func, lik_func, dataset.num_train, inducing_inputs, flags)
    return gp, hyper_params


def get_optimizer(flags, global_step=None):
    """Construct the optimizer from the information in the flags

    If the global step is not given, then a function is returned that takes the global step as a
    parameter and updates the learning rate.

    Args:
        flags: dictionary with parameters
        global_step: (optional) the step in training
    Returns:
        the optimizer and a function to update the learning rate
    """
    schedule = [flags['lr'], flags['lr'] * flags['lr_drop_factor']]
    drop_steps = flags['lr_drop_steps']

    if global_step is None:
        learning_rate = tf.get_variable("lr", initializer=tf.constant(flags['lr']), trainable=False)

        def _update_learning_rate(step):
            if drop_steps > 0:
                learning_rate.assign(tf.train.piecewise_constant(step, [drop_steps], schedule))

        return getattr(tf.train, flags['optimizer'])(learning_rate), _update_learning_rate

    if drop_steps > 0:
        learning_rate = tf.train.piecewise_constant(global_step, [drop_steps], schedule)
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

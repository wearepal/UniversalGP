"""Functions for helping with training"""
from pathlib import Path
import numpy as np
import tensorflow as tf

from . import plot
from .. import cov, inf, lik


def construct_from_flags(flags, input_dim, output_dim, liklihood_name, inducing_inputs, num_train):
    """Construct a GP model with the given parameters

    Args:
        flags: dictionary with parameters
        input_dim: input dimension
        output_dim: output dimension
        liklihood_name: a string that names a liklihood function
        inducing_inputs: inducing inputs
        num_train: number of training examples
    Returns:
        a GP object and the hyper parameters
    """
    cov_func = [getattr(cov, flags['cov'])(input_dim, flags) for _ in range(output_dim)]
    lik_func = getattr(lik, liklihood_name)(flags)
    hyper_params = lik_func.get_params() + sum([k.get_params() for k in cov_func], [])

    gp = getattr(inf, flags['inf'])(cov_func, lik_func, num_train, inducing_inputs, flags)
    return gp, hyper_params, getattr(tf.train, flags['optimizer'])(flags['lr'])


def post_training(pred_mean, pred_var, out_dir, dataset, args):
    """Call all functions that need to be executed after training has finished

    Args:
        pred_mean: predicted mean
        pred_var: predicted variance
        out_dir: path where to store predictions or None
        dataset: dataset object
        args: additional arguments
    """
    working_dir = Path(out_dir) if args['save_dir'] else Path(".")
    with open(working_dir / Path(f"flag_{args['model_name']}.txt"), 'w') as f:
        f.write("\n".join([f"--{k}={v}" for k, v in args.items()]))
    if args['preds_path']:
        np.savez_compressed(working_dir / Path(args['preds_path']),
                            pred_mean=pred_mean, pred_var=pred_var)
    if args['plot']:
        getattr(plot, args['plot'])(pred_mean, pred_var, dataset)

"""Load a model from a checkpoint"""
import sys
import tensorflow as tf

# importing is difficult because it depends on how the program is started
sys.path.append('..')
from universalgp import inf, datasets


def load(checkpoint_path, flags, input_dim, output_dim, num_train, num_inducing, inf_name, lik_name,
         cov_name):
    """Load a model from a checkpoint

    First train the model like this:

    ```
    python gaussian_process.py --save_dir=/its/home/tk324/tensorflow --model_name=m1
    ```

    You can of course also choose a different directory and model name.

    Args:
        checkpoint_path: path to the checkpoint file
        config: dictionary that contains all configuration
    Returns:
        Gaussian Process
    """
    gp = getattr(inf, inf_name)(
        {
            'num_components': flags.get('num_components', None),
            'diag_post': flags.get('diag_post', None),
            'iso': flags['iso'],
            'num_samples_pred': flags.get('num_samples_pred', None),
            'cov': cov_name,
        },
        lik_name,
        output_dim,
        num_train,
        num_inducing
    )
    # load GP model from checkpoint
    checkpoint = tf.train.Checkpoint(gp=gp)
    checkpoint.restore(checkpoint_path)
    gp.build((num_train, input_dim))
    return gp


def parse_and_load(checkpoint_path, dataset_name, inf_name, cov_name, flags):
    """Parse the config and then load a model from a checkpoint"""
    dataset = get_dataset(dataset_name, flags)
    gp = load(checkpoint_path, flags, dataset.input_dim, dataset.output_dim, dataset.num_train,
              dataset.inducing_inputs.shape[0], inf_name, dataset.lik, cov_name)
    return gp, dataset


def get_dataset(dataset_name, flags):
    """Get a dataset by name"""
    return getattr(datasets, dataset_name)(flags)

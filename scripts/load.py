"""Load a model from a checkpoint"""
import sys
from tensorflow.contrib import eager as tfe

# importing is difficult because it depends on how the program is started
# we try different methods here and hope that one works:
sys.path.append('..')
from universalgp import inf, cov, lik
import datasets


def load(checkpoint_path, config):
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
    tfe.enable_eager_execution()

    # load GP model from checkpoint
    with tfe.restore_variables_on_create(checkpoint_path):
        gp = config['inf']([config['cov'](config['input_dim'], {'iso': config['iso']})
                            for _ in range(config['output_dim'])],
                           config['lik']({'num_samples_pred': config.get('num_samples_pred', None)}),
                           config['num_train'],
                           config['num_inducing'],
                           {'num_components': config.get('num_components', None),
                            'diag_post': config.get('diag_post', None)})
    return gp


def parse_and_load(checkpoint_path, dataset_name, inf_name, cov_name, flags):
    """Parse the config and then load a model from a checkpoint"""
    dataset = getattr(datasets, dataset_name)()
    gp = load(checkpoint_path,
              dict(input_dim=dataset.input_dim, output_dim=dataset.output_dim, num_train=dataset.num_train,
                   num_inducing=dataset.inducing_inputs.shape[0], lik=getattr(lik, dataset.lik),
                   cov=getattr(cov, cov_name), inf=getattr(inf, inf_name), **flags))
    return gp, dataset

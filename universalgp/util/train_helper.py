"""Functions for loading"""
from .. import cov, inf, lik


def construct_gp(flags, input_dim, output_dim, liklihood_name, inducing_inputs, num_train):
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
    return gp, hyper_params

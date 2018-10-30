import numpy as np

# importing is difficult because it depends on how the program is started
# we try different methods here and hope that one works:
try:
    from scripts.load import parse_and_load
except ModuleNotFoundError:
    from load import parse_and_load

COV = 'SquaredExponential'
INF = 'Exact'
FLAGS = {
    'num_components': 1,
    'num_samples_pred': 2000,
    'diag_post': False,
    'iso': False,
    'num_inducing': 50,
}
DATASET = 'simple_example'
# CHECKPOINT_PATH = "/its/home/tk324/tensorflow/sm1/model.ckpt-500"
CHECKPOINT_PATH = "/home/ubuntu/out/m1/model.ckpt-500"
SAVE_PATH = "./vars.npz"


def main():
    # load GP model from checkpoint
    gp, _ = parse_and_load(CHECKPOINT_PATH, DATASET, INF, COV, FLAGS)

    # get variables
    var_collection = {var.name: var.numpy() for var in gp.variables}

    # save in file
    np.savez_compressed(SAVE_PATH, **var_collection)

    ##### To load the contents from the file again:
    # vars = np.load("vars.npz")
    # print(vars.keys())
    # print(vars['variational_inference/means:0'])


if __name__ == "__main__":
    main()

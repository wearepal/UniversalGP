import numpy as np

# importing is difficult because it depends on how the program is started
# we try different methods here and hope that one works:
try:
    from scripts.load import parse_and_load
except ModuleNotFoundError:
    from load import parse_and_load

COV = 'SquaredExponential'
INF = 'Variational'
FLAGS = {
    'num_components': 1,
    'num_samples_pred': 2000,
    'diag_post': False,
    'iso': False
}
DATASET = 'sensitive_example'
CHECKPOINT_PATH = "/its/home/tk324/tensorflow/m1/ckpt-504"
AS_TXT_FILE = True
RESULT_PATH = "./predictions"  # without file ending


def main():
    # load GP model from checkpoint
    gp, dataset = parse_and_load(CHECKPOINT_PATH, DATASET, INF, COV, FLAGS)

    # make predictions
    pred_mean, pred_var = gp.predict({'input': dataset.xtest})

    # save in file
    if AS_TXT_FILE:
        full_path = RESULT_PATH + ".txt"
        np.savetxt(full_path, np.column_stack((dataset.xtest[:, 0], pred_mean.numpy()[:, 0], pred_var.numpy()[:, 0])),
                   header='xtest, pred_mean, pred_var')
    else:
        np.savez_compressed(RESULT_PATH, pred_mean=pred_mean, pred_var=pred_var)

    ##### To load the contents from the file again:
    # xtest, pred_means, pred_vars = np.loadtxt("predictions.txt", unpack=True)


if __name__ == "__main__":
    main()

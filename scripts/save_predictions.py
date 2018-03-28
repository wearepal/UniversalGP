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
RESULT_PATH = "./predictions.npz"


def main():
    # load GP model from checkpoint
    gp, dataset = parse_and_load(CHECKPOINT_PATH, DATASET, INF, COV, FLAGS)

    # pred_means, pred_vars = gp.predict([[1., 1.]])
    # print(pred_means)

    # make predictions
    pred_means, pred_vars = gp.predict(dataset.xtest)
    # save in file
    np.savez_compressed(RESULT_PATH, **{'pred_means': pred_means, 'pred_vars': pred_vars})

    ##### To load the file again:
    # loaded_data = np.load("predictions.npz")
    # pred_means = loaded_data['pred_means']

if __name__ == "__main__":
    main()

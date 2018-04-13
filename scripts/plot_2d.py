"""
Plot predicted data from a text file
"""

import numpy as np
from matplotlib import pyplot as plt

# importing is difficult because it depends on how the program is started
# we try different methods here and hope that one works:
try:
    from scripts.load import get_dataset
except ModuleNotFoundError:
    from load import get_dataset

INPUT_PATH = "predictions.npz"
AS_TXT_FILE = False  # whether the data is stored as a text file
DATASET = "maize_yield"
IN_DIM = 0


def main():
    """Load and plot"""
    if AS_TXT_FILE:
        # we assume the columns in the file are: xtest, pred_mean, pred_var
        xtest, pred_mean, pred_var = np.loadtxt(INPUT_PATH, unpack=True)
    else:
        preds = np.load(INPUT_PATH)
        pred_mean, pred_var = preds['pred_mean'], preds['pred_var']
        dataset = get_dataset(DATASET)
        xtest = dataset.xtest[IN_DIM]

    sorted_index = np.argsort(xtest)
    xtest = xtest[sorted_index]
    pred_mean = pred_mean[sorted_index]
    pred_var = pred_var[sorted_index]

    plt.plot(xtest, pred_mean, 'x', mew=2, label='predictions')

    upper_bound = pred_mean + 1.96 * np.sqrt(pred_var)
    lower_bound = pred_mean - 1.96 * np.sqrt(pred_var)

    plt.fill_between(xtest, lower_bound, upper_bound, color='gray', alpha=0.25, label='95% CI')
    plt.legend(loc='lower left')
    plt.show()


if __name__ == "__main__":
    main()

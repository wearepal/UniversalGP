import numpy as np
from matplotlib import pyplot as plt


def simple_1d(pred_mean, pred_var, xtrain, ytrain, xtest, ytest, in_dim=0):
    """Plot train and test data and predicted data with uncertainty."""
    out_dims = len(ytrain[0])
    in_dim = 0
    for i in range(out_dims):
        plt.subplot(out_dims, 1, i + 1)
        plt.plot(xtrain[:, in_dim], ytrain[:, i], '.', mew=2, label='trainings')
        plt.plot(xtest[:, in_dim], ytest[:, i], 'o', mew=2, label='tests')
        plt.plot(xtest[:, in_dim], pred_mean[:, i], 'x', mew=2, label='predictions')

        upper_bound = pred_mean[:, i] + 1.96 * np.sqrt(pred_var[:, i])
        lower_bound = pred_mean[:, i] - 1.96 * np.sqrt(pred_var[:, i])

        plt.fill_between(xtest[:, in_dim], lower_bound, upper_bound, color='gray', alpha=0.25, label='95% CI')
    plt.legend(loc='lower left')
    plt.show()

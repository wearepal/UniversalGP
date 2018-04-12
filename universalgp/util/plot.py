import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from .plot_classification import Classification


def simple_1d(pred_mean, pred_var, data, in_dim=0):
    """Plot train and test data and predicted data with uncertainty."""
    flexible_1d(data.xtest, (pred_mean, pred_var), (data.xtrain, data.ytrain), (data.xtest, data.ytest), in_dim)


def flexible_1d(xpreds, preds, train, test, in_dim=0):
    """Plot train and test data and predicted data with uncertainty.

    Args:
        xpreds: inputs for predictions
        preds: predictions
        train: training inputs and outputs
        test: testing inputs and outputs
        in_dim: (optional) the input dimension that will be plotted
    """
    xtrain, ytrain = train
    xtest, ytest = test
    pred_mean, pred_var = preds
    sorted_index = np.argsort(xpreds[:, 0])
    xpreds = xpreds[sorted_index]
    pred_mean = pred_mean[sorted_index]
    pred_var = pred_var[sorted_index]
    out_dims = len(ytrain[0])
    for i in range(out_dims):
        plt.subplot(out_dims, 1, i + 1)
        plt.plot(xtrain[:, in_dim], ytrain[:, i], '.', mew=2, label='trainings')
        plt.plot(xtest[:, in_dim], ytest[:, i], 'o', mew=2, label='tests')
        plt.plot(xpreds[:, in_dim], pred_mean[:, i], 'x', mew=2, label='predictions')

        upper_bound = pred_mean[:, i] + 1.96 * np.sqrt(pred_var[:, i])
        lower_bound = pred_mean[:, i] - 1.96 * np.sqrt(pred_var[:, i])

        plt.fill_between(xpreds[:, in_dim], lower_bound, upper_bound, color='gray', alpha=0.25, label='95% CI')
    plt.legend(loc='lower left')
    plt.show()


def simple_2d(pred_mean, pred_var, data, in_dim_a=0, in_dim_b=1, save_animation=False):
    """Plot train and test data and predicted data with uncertainty."""
    xtrain, ytrain, xtest, ytest = data.xtrain, data.ytrain, data.xtest, data.ytest
    out_dims = len(ytrain[0])
    fig = plt.figure()
    for i in range(out_dims):
        ax = fig.add_subplot(out_dims, 1, i + 1, projection='3d')
        ax.scatter(xtrain[:, in_dim_a], xtrain[:, in_dim_b], ytrain[:, i], marker='.', s=40, label='trainings')
        ax.scatter(xtest[:, in_dim_a], xtest[:, in_dim_b], ytest[:, i], marker='o', s=40, label='tests')
        ax.scatter(xtest[:, in_dim_a], xtest[:, in_dim_b], pred_mean[:, i], marker='x', s=40, label='predictions')

        upper_bound = pred_mean[:, i] + 1.96 * np.sqrt(pred_var[:, i])
        lower_bound = pred_mean[:, i] - 1.96 * np.sqrt(pred_var[:, i])

        ax.add_collection3d(plt.fill_between(xtest[:, in_dim_a], lower_bound, upper_bound, color='gray', alpha=0.25,
                                             label='95% CI'), zs=xtest[:, in_dim_b], zdir='y')
    # legend doesn't work with `add_collection3d`
    # plt.legend(loc='lower left')

    if save_animation:
        def _rotate(angle):
            ax.view_init(azim=angle)

        rot_animation = animation.FuncAnimation(fig, _rotate, frames=np.arange(0, 362, 2), interval=100)
        rot_animation.save('rotation.gif', dpi=80, writer='imagemagick')
    else:
        plt.show()


def classification_2d(pred, _, data):
    if data.lik == "LikelihoodLogistic":
        pred = (pred > 0.5).astype(np.float)
    elif data.lik == "LikelihoodSoftmax":
        pred = np.argmax(pred, axis=1)

    group = Classification(data.xtest, data.ytest)
    group.plot_2d_prediction(pred)


# def classification_2d_sensitive(pred, sensi_attr_test, pred_var, xtrain, ytrain, xtest, ytest):
#     group = Classification(xtest, ytest)
#     group.plot_2d_sensitive_prediction(pred, sensi_attr_test)

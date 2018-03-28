"""Plot train and test data and predicted data with uncertainty."""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

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
DATASET = 'maize_yield'
CHECKPOINT_PATH = "/its/home/tk324/tensorflow/maize1/model.ckpt-500"
SAVE_ANIMATION = False
IN_DIM_A = 0
IN_DIM_B = 2


def main():
    # load GP model from checkpoint
    gp, dataset = parse_and_load(CHECKPOINT_PATH, DATASET, INF, COV, FLAGS)
    xtrain, ytrain = dataset.xtrain, dataset.ytrain
    xtest, ytest = dataset.xtest, dataset.ytest
    pred_mean, pred_var = gp.predict(xtest)

    out_dims = dataset.output_dim
    fig = plt.figure()
    for i in range(out_dims):
        ax = fig.add_subplot(out_dims, 1, i + 1, projection='3d')
        ax.scatter(xtrain[:, IN_DIM_A], xtrain[:, IN_DIM_B], ytrain[:, i], marker='.', s=40, label='trainings')
        ax.scatter(xtest[:, IN_DIM_A], xtest[:, IN_DIM_B], ytest[:, i], marker='o', s=40, label='tests')
        ax.scatter(xtest[:, IN_DIM_A], xtest[:, IN_DIM_B], pred_mean[:, i], marker='x', s=40, label='predictions')

        upper_bound = pred_mean[:, i] + 1.96 * np.sqrt(pred_var[:, i])
        lower_bound = pred_mean[:, i] - 1.96 * np.sqrt(pred_var[:, i])

        ax.add_collection3d(plt.fill_between(xtest[:, IN_DIM_A], lower_bound, upper_bound, color='gray', alpha=0.25,
                                             label='95% CI'), zs=xtest[:, IN_DIM_B], zdir='y')
    # legend doesn't work with `add_collection3d`
    # plt.legend(loc='lower left')

    if SAVE_ANIMATION:
        def _rotate(angle):
            ax.view_init(azim=angle)

        rot_animation = animation.FuncAnimation(fig, _rotate, frames=np.arange(0, 362, 2), interval=100)
        rot_animation.save('rotation.gif', dpi=80, writer='imagemagick')
    else:
        plt.show()

if __name__ == "__main__":
    main()

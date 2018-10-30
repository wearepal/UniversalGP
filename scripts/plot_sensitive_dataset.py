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

MARKER = ['x', 'o']
COLOR = ['b', 'r', 'g', 'c', 'm', 'y', 'k']

INPUT_PATH = "predictions.npz"
DATASET = 'sensitive_example'
FLAGS = dict(
    num_inducing=50,
)
NUM_TO_DRAW = 1000


def main():
    dataset = get_dataset(DATASET, FLAGS)
    sensi_attr = dataset.strain
    X, y = dataset.xtrain, dataset.ytrain
    y = np.squeeze(y)
    sensi_attr = np.squeeze(sensi_attr)

    ### Calculate bias
    biased_acceptance1 = np.sum(y[sensi_attr == 0] == 1) / np.sum(sensi_attr == 0)
    biased_acceptance2 = np.sum(y[sensi_attr == 1] == 1) / np.sum(sensi_attr == 1)
    print(f"P(y=1|s=0) = {biased_acceptance1}, P(y=1|s=1) = {biased_acceptance2}")

    if X.shape[0] < NUM_TO_DRAW:
        x_draw = X
        y_draw = y
        s_draw = sensi_attr
    else:
        x_draw = X[:NUM_TO_DRAW]
        y_draw = y[:NUM_TO_DRAW]
        s_draw = sensi_attr[:NUM_TO_DRAW]

    class_label = list(set(y_draw))
    sensitive_label = list(set(s_draw))

    plt.figure()
    for i in sensitive_label:
        X_s = x_draw[s_draw == i]
        y_s = y_draw[s_draw == i]
        sensitive_name = f"Sensi-Label={int(i)},"

        for j in class_label:
            class_name = f" Class-Label={int(j)}"
            label = sensitive_name + class_name
            plt.scatter(X_s[y_s == j][:, 0], X_s[y_s == j][:, 1], color=COLOR[int(j)],
                        marker=MARKER[int(i)], label=label)

    # plt.tick_params(axis='x', which='both', bottom='off', top='off',
    #                 labelbottom='off')  # dont need the ticks to see the data distribution
    # plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
    plt.grid()
    plt.legend(loc="best")
    plt.show()


if __name__ == "__main__":
    main()

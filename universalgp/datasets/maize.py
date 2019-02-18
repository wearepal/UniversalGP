"""Maize dataset"""
from pathlib import Path
import numpy as np
from scipy.stats import zscore

from .definition import Dataset, to_tf_dataset

DATA_PATH = Path("universalgp") / Path("datasets") / Path("data") / Path("Maize Yield150318.csv")


def maize_yield(_):
    """Maize dataset"""
    data = np.loadtxt(DATA_PATH, delimiter=',', skiprows=1).astype(np.float32)
    x = data[:, 1:]
    y = zscore(data[:, :1])  # standardize the output, otherwise the GP will have problems
    itest = 5  # index of the test point
    xtrain = np.concatenate((x[:itest], x[itest + 1:]))
    ytrain = np.concatenate((y[:itest], y[itest + 1:]))
    xtest = x[itest:itest + 1]
    ytest = y[itest:itest + 1]

    return Dataset(
        input_dim=8,
        output_dim=1,
        train_fn=to_tf_dataset(xtrain, ytrain),
        test_fn=to_tf_dataset(xtest, ytest),
        inducing_inputs=xtrain,
        num_train=len(ytrain),
        lik="LikelihoodGaussian",
        metric="RMSE",
        xtrain=xtrain,
        ytrain=ytrain,
        xtest=xtest,
        ytest=ytest,
    )

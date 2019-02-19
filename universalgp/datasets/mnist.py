"""
MNIST dataset
"""
import tensorflow_datasets as tfds

import sklearn.cluster

from .definition import Dataset

NUM_INDUCING = 100


def _init_z(train_inputs, num_inducing):
    # Initialize inducing points using clustering.
    print('start clustering')
    mini_batch = sklearn.cluster.MiniBatchKMeans(num_inducing)
    mini_batch.fit_predict(train_inputs)
    inducing_locations = mini_batch.cluster_centers_
    print('done clustering')
    return inducing_locations


def mnist(_):
    """MNIST dataset with one hot labels"""
    ds_train, ds_test = tfds.load(name="mnist", split=["train", "test"])

    return Dataset(
        input_dim=28 * 28,
        output_dim=10,
        train=ds_train,
        test=ds_test,
        inducing_inputs=_init_z(data.train.images, NUM_INDUCING),
        num_train=data.train.num_examples,
        lik="LikelihoodSoftmax",
        metric="soft_accuracy",
    )

"""
MNIST dataset
"""
from tensorflow.examples.tutorials.mnist import input_data

import sklearn.cluster

from .definition import Dataset, to_tf_dataset_fn

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
    data = input_data.read_data_sets('./universalgp/datasets/data/', one_hot=True)

    return Dataset(
        input_dim=28 * 28,
        output_dim=10,
        train_fn=to_tf_dataset_fn(data.train.images, data.train.labels),
        test_fn=to_tf_dataset_fn(data.test.images, data.test.labels),
        inducing_inputs=_init_z(data.train.images, NUM_INDUCING),
        num_train=data.train.num_examples,
        lik="LikelihoodSoftmax",
        metric="soft_accuracy",
    )

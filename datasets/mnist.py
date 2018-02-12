"""
MNIST dataset
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import sklearn.cluster

NUM_INDUCING = 100


def _init_z(train_inputs, num_inducing):
    # Initialize inducing points using clustering.
    mini_batch = sklearn.cluster.MiniBatchKMeans(num_inducing)
    mini_batch.fit_predict(train_inputs)
    inducing_locations = mini_batch.cluster_centers_
    return inducing_locations


def mnist():
    mnist = input_data.read_data_sets('./datasets/data/', one_hot=True)

    return {'input_dim': 28 * 28,
            'output_dim': 10,
            'train_fn': lambda: tf.data.Dataset.from_tensor_slices(({'input': mnist.train.images}, mnist.train.labels)),
            'test_fn': lambda: tf.data.Dataset.from_tensor_slices(({'input': mnist.test.images}, mnist.test.labels)),
            'inducing_inputs': _init_z(mnist.train.images, NUM_INDUCING),
            'num_train': mnist.train.num_examples,
           }

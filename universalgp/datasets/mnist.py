"""
MNIST dataset
"""
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import sklearn.cluster

from .definition import Dataset

NUM_INDUCING = 100
IMG_SIZE = 28 * 28
NUM_CLASSES = 10


def _init_z(train_inputs):
    # Initialize inducing points using clustering.
    print('start clustering')
    mini_batch = sklearn.cluster.MiniBatchKMeans(NUM_INDUCING)
    mini_batch.fit_predict(np.reshape(train_inputs, (-1, IMG_SIZE)))
    inducing_locations = mini_batch.cluster_centers_
    print('done clustering')
    return inducing_locations


def _convert_examples(features):
    inputs = tf.image.convert_image_dtype(tf.reshape(features['image'], [IMG_SIZE]), tf.float32)
    return {'input': inputs}, tf.one_hot(features['label'], NUM_CLASSES)


def mnist(_):
    """MNIST dataset with one hot labels"""
    ds_train, ds_test = tfds.load(name="mnist", split=["train", "test"])
    numpy_train = tfds.as_numpy(tfds.load(name="mnist", split="train", batch_size=-1))

    return Dataset(
        input_dim=IMG_SIZE,
        output_dim=NUM_CLASSES,
        train=ds_train.map(_convert_examples),
        test=ds_test.map(_convert_examples),
        inducing_inputs=_init_z(numpy_train['image']),
        num_train=len(numpy_train['image']),
        lik="LikelihoodSoftmax",
        metric="soft_accuracy",
    )

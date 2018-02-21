import numpy as np
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels
from tensorflow.python.framework import dtypes

import sklearn.cluster

from universalgp.datasets import DataSet
import universalgp
from universalgp import lik, cov, inf, util


FLAGS = util.util.get_flags()
BATCH_SIZE = FLAGS.batch_size
LEARNING_RATE = FLAGS.learning_rate
DISPLAY_STEP = FLAGS.display_step
EPOCHS = FLAGS.n_epochs
NUM_SAMPLES = FLAGS.mc_train
NUM_INDUCING = FLAGS.n_inducing
IS_ARD = FLAGS.is_ard
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
ONE_HOT = True
TRAIN_DIR = 'experiments/data/MNIST_data'


def init_z(train_inputs, num_inducing):
    # Initialize inducing points using clustering.
    mini_batch = sklearn.cluster.MiniBatchKMeans(num_inducing)
    cluster_indices = mini_batch.fit_predict(train_inputs)
    inducing_locations = mini_batch.cluster_centers_
    return inducing_locations


def main():
    data, test, _ = import_mnist()

    # Setup initial values for the model.
    likelihood = lik.LikelihoodSoftmax()
    kern = [cov.SquaredExponential(data.X.shape[1], length_scale=10, sf=1, iso=IS_ARD) for _ in range(10)]
    # kern = [kernels.ArcCosine(X.shape[1], 2, 3, 5.0, 1.0, input_scaling=True) for i in range(10)]
    # RadialBasis(X.shape[1], input_scaling=True) for i in xrange(10)]

    Z = init_z(data.X, NUM_INDUCING)
    vinf = inf.Variational(kern, likelihood, num_samples=NUM_SAMPLES)
    m = universalgp.GaussianProcess(Z, kern, vinf, likelihood)

    # setting up loss to be reported during training
    # error_rate = losses.ZeroOneLoss(data.Dout)

    import time
    otime = time.time()
    o = tf.train.RMSPropOptimizer(LEARNING_RATE)
    start = time.time()
    m.fit(data, o, loo_steps=50, var_steps=50, epochs=EPOCHS,
          batch_size=BATCH_SIZE, display_step=DISPLAY_STEP
          # ,loss=error_rate)
         )
    print(time.time() - start)
    print(time.time() - otime)

    ypred = m.predict(data, test.X)[0]
    # print(f"Final {error_rate.get_name()}={error_rate.eval(test.Y, ypred):.4}")
    print(f"ypred={ypred}")


def process_mnist(images, dtype = dtypes.float32, reshape=True):
    if reshape:
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])
    if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)

    return images


def get_data_info(images):
    _, cols = images.shape
    std = np.zeros(cols)
    mean = np.zeros(cols)
    for col in range(cols):
        std[col] = np.std(images[:, col])
        mean[col] = np.mean(images[:, col])
    return mean, std

def standardize_data(images, means, stds):
    data = images.copy()
    _, cols = data.shape
    for col in range(cols):
        if stds[col] == 0:
            data[:, col] = (data[:, col] - means[col])
        else:
            data[:, col] = (data[:, col] - means[col]) / stds[col]
    return data


def import_mnist(validation_size=0):
    """
    This import mnist and saves the data as an object of our DataSet class
    :param concat_val: Concatenate training and validation
    :return:
    """

    local_file = base.maybe_download(TRAIN_IMAGES, TRAIN_DIR, SOURCE_URL + TRAIN_IMAGES)
    with open(local_file, 'rb') as f:
        train_images = extract_images(f)

    local_file = base.maybe_download(TRAIN_LABELS, TRAIN_DIR, SOURCE_URL + TRAIN_LABELS)
    with open(local_file, 'rb') as f:
        train_labels = extract_labels(f, one_hot=ONE_HOT)

    local_file = base.maybe_download(TEST_IMAGES, TRAIN_DIR, SOURCE_URL + TEST_IMAGES)
    with open(local_file, 'rb') as f:
        test_images = extract_images(f)

    local_file = base.maybe_download(TEST_LABELS, TRAIN_DIR, SOURCE_URL + TEST_LABELS)
    with open(local_file, 'rb') as f:
        test_labels = extract_labels(f, one_hot=ONE_HOT)

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    # process images
    train_images = process_mnist(train_images)
    validation_images = process_mnist(validation_images)
    test_images = process_mnist(test_images)

    # standardize data
    train_mean, train_std = get_data_info(train_images)
    train_images = standardize_data(train_images, train_mean, train_std)
    validation_images = standardize_data(validation_images, train_mean, train_std)
    test_images = standardize_data(test_images, train_mean, train_std)

    data = DataSet(train_images, train_labels)
    test = DataSet(test_images, test_labels)
    val = DataSet(validation_images, validation_labels)

    return data, test, val


if __name__ == '__main__':
    main()

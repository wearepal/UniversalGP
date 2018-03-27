#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thursday March 22 13:41:54 2018

Usage: Generate the simple synthetic data with two non-sensitive features and one sensitive feature.
       A sensitive feature, 0.0 : protected group (e.g., female)
                            1.0 : non-protected group (e.g., male).
       For parity demographic
"""

from random import seed, shuffle
import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal

from .definition import Dataset


SEED = 123
seed(SEED)  # set the random seed, which can be reproduced again
np.random.seed(SEED)


def sensitive_example():
    """Simple example with synthetic data."""
    n_all = 200
    disc_factor = np.pi / 5.0  # discrimination in the data -- decraese it to generate more discrimination
    inputs, outputs, sensi_attr = _generate_feature(n_all, disc_factor)

    num_train = 150
    xtrain, ytrain, xtest, ytest, sensi_attr_train, sensi_attr_test = _select_training_and_test(
        inputs, outputs[..., np.newaxis], sensi_attr, num_train)
    num_inducing = 150

    return Dataset(
        train_fn=lambda: tf.data.Dataset.from_tensor_slices(({'input':_const(xtrain)}, _const(ytrain))),
        test_fn=lambda: tf.data.Dataset.from_tensor_slices(({'input':_const(xtest)}, _const(ytest))),
        num_train=num_train,
        input_dim=2,
        inducing_inputs=xtrain[::num_train // num_inducing],
        output_dim=1,
        lik="LikelihoodLogistic",
        metric="logistic_accuracy",
        train_feature_columns=[tf.feature_column.numeric_column('input', shape=2)],
        test_feature_columns=[tf.feature_column.numeric_column('input', shape=2)],
        xtrain=xtrain,
        ytrain=ytrain,
        xtest=xtest,
        ytest=ytest,
        strain=sensi_attr_train,
        stest=sensi_attr_test
    )


def _gaussian_generator(mean, cov, label, n):
    distribution = multivariate_normal(mean=mean, cov=cov)
    X = distribution.rvs(n)
    y = np.ones(n, dtype=float) * label
    return distribution, X, y


def _generate_feature(n, disc_factor):
    """Generate the non-sensitive features randomly"""
    mu1, sigma1 = [2, 2], [[5, 1], [1, 5]]
    mu2, sigma2 = [-2, -2], [[10, 1], [1, 3]]
    nv1, X1, y1 = _gaussian_generator(mu1, sigma1, 1, n)  # positive class
    nv2, X2, y2 = _gaussian_generator(mu2, sigma2, 0, n)  # negative class

    # join the positive and negative class clusters
    inputs = np.vstack((X1, X2))
    outputs = np.hstack((y1, y2))

    rotation = np.array([[np.cos(disc_factor), -np.sin(disc_factor)],
                         [np.sin(disc_factor), np.cos(disc_factor)]])
    inputs_aux = np.dot(inputs, rotation)

    #### Generate the sensitive feature here ####
    sensi_attr = []  # this array holds the sensitive feature value
    for i in range(len(inputs)):
        x = inputs_aux[i]

        # probability for each cluster that the point belongs to it
        p1 = nv1.pdf(x)
        p2 = nv2.pdf(x)

        # normalize the probabilities from 0 to 1
        s = p1 + p2
        p1 = p1 / s
        p2 = p2 / s

        r = np.random.uniform()  # generate a random number from 0 to 1

        if r < p1:  # the first cluster is the positive class
            sensi_attr.append(1.0)  # 1.0 means its male
        else:
            sensi_attr.append(0.0)  # 0.0 -> female

    sensi_attr = np.array(sensi_attr)

    return inputs, outputs, sensi_attr


def _const(arr):
    return tf.constant(arr, dtype=tf.float32)


def _select_training_and_test(inputs, outputs, sensi_attr, num_train):
    idx = np.arange(len(inputs))
    np.random.shuffle(idx)
    num_train = 2 * num_train
    xtrain = inputs[idx[:num_train]]
    ytrain = outputs[idx[:num_train]]
    sensi_attr_train = sensi_attr[idx[:num_train]]

    xtest = inputs[idx[num_train:]]
    ytest = outputs[idx[num_train:]]
    sensi_attr_test = sensi_attr[idx[num_train:]]

    return xtrain, ytrain, xtest, ytest, sensi_attr_train, sensi_attr_test

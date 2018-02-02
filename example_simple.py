#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 12:41:54 2018

@author: zc223
"""

import universalgp
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Generate synthetic data.
N_all = 200
N = 50
inputs = 5 * np.linspace(0, 1, num=N_all)[:, np.newaxis]
outputs = np.cos(inputs)

# selects training and test
idx = np.arange(N_all)
np.random.shuffle(idx)
xtrain = inputs[idx[:N]]
ytrain = outputs[idx[:N]]
train_data = universalgp.datasets.DataSet(xtrain, ytrain)
xtest = inputs[idx[N:]]
ytest = outputs[idx[N:]]

# Initialize the Gaussian process.
lik = universalgp.lik.LikelihoodGaussian()
cov = universalgp.cov.SquaredExponential(1)

# mean = universalgp.mean.ZeroOffset()
# inf = universalgp.inf.Variational(cov, lik)
inf = universalgp.inf.Exact(cov, lik)

inducing_inputs = xtrain
model = universalgp.GaussianProcess(inducing_inputs, cov, inf, lik)

# Train the model.
optimizer = tf.train.RMSPropOptimizer(0.005)
model.fit(train_data, optimizer, batch_size=1, var_steps=10, epochs=100, display_step=10)

# Predict new inputs.
ypred, _ = model.predict(train_data, xtest)
plt.plot(xtrain, ytrain, '.', mew=2)
plt.plot(xtest, ytest, 'o', mew=2)
plt.plot(xtest, ypred, 'x', mew=2)
plt.show()

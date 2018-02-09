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
xtest = inputs[np.sort(idx[N:])]
ytest = outputs[np.sort(idx[N:])]

# Initialize the Gaussian process.
lik = universalgp.lik.LikelihoodGaussian()
cov = [universalgp.cov.SquaredExponential(1)]

# mean = universalgp.mean.ZeroOffset()
inf = universalgp.inf.Variational(cov, lik)
# inf = universalgp.inf.Exact(cov, lik)

inducing_inputs = xtrain
model = universalgp.GaussianProcess(inducing_inputs, cov, inf, lik)

# Train the model.
optimizer = tf.train.RMSPropOptimizer(0.005)
model.fit(train_data, optimizer, var_steps=1, epochs=500, display_step=10)

# Predict new inputs.
pred_mean, pred_var = model.predict(train_data, xtest)
plt.plot(xtrain, ytrain, '.', mew=2, label='trainings')
plt.plot(xtest, ytest, 'o', mew=2, label='tests')
plt.plot(xtest, pred_mean, 'x', mew=2, label='predictions')

upper_bound = pred_mean + 1.96 * np.sqrt(pred_var)
lower_bound = pred_mean - 1.96 * np.sqrt(pred_var)

plt.fill_between(np.squeeze(xtest), lower_bound, upper_bound, color='gray', alpha=0.25,
                 label='95% CI')
plt.legend(loc='lower left')
# plt.plot(xtest, upper_bound, '--', c='gray', lw=0.5)
# plt.plot(xtest, lower_bound, '--', c='gray', lw=0.5)
plt.show()

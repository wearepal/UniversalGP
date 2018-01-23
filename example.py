#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 16:11:52 2018

"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def covariance_func(length):
  def f(x, y):
    return tf.exp(-.5*(x.reshape((-1, 1)) - y.reshape((1, -1)))**2 / length**2)
  return f


def main():
  # parameters
  num_functions_to_sample = 3
  num_samples_per_function = 100
  noisy_std = 0.05

  # sample locations
  sample_points = np.linspace(0, 1, num_samples_per_function)
  # sample_points = np.array([0.5])

  # get covariance function
  cov = covariance_func(length=0.2)
  # compute covariance of the sample locations
  cov_samples = cov(sample_points, sample_points)

  # prior
  # mean = np.zeros(100)
  # f1, f2, f3 = np.random.multivariate_normal(mean, cov_samples,
  #                                            num_functions_to_sample)

  # data points
  x = np.array([0.1, 0.3, 0.8])
  y = np.array([0., 0.2, -0.5])

  # compute mean and covariance conditioned on the data points
  cov_mix = cov(sample_points, x)
  precision_data = tf.linalg.inv(cov(x, x) + noisy_std**2 * np.eye(len(x)))
  mean_cond = cov_mix @ precision_data @ y.T
  cov_cond = cov_samples - cov_mix @ precision_data @ cov_mix.T

  # draw sample functions and plot them
  f1, f2, f3 = np.random.multivariate_normal(mean_cond, cov_cond,
                                             num_functions_to_sample)
  plt.plot(sample_points, f1)
  plt.plot(sample_points, f2)
  plt.plot(sample_points, f3)

  # plot data points
  plt.plot(x, y, 'k+')

  # plot uncertainty region
  std = np.sqrt(cov_cond.diagonal())
  plt.fill_between(sample_points, mean_cond - 2 * std, mean_cond + 2 * std,
                   facecolor='lightgray')

  plt.show()


if __name__ == "__main__":
  main()

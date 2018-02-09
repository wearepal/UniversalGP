#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 14:56:13 2018

@author: zc223
"""

import numpy as np
import scipy.linalg as sl
import scipy.special


class Normal(object):
    def __init__(self, mean, covar):
        self.mean = mean
        self.covar = covar


class CholNormal(Normal):
    def prob(self, val):
        return np.exp(self.log_prob(val))

    def log_prob(self, val):
        dim = np.shape(self.mean)[0] + 0.0
        diff = np.expand_dims(val - self.mean, 1)
        chol = self.covar

        quad_form = np.sum(diff * sl.solve_triangular(chol, diff, lower=True))
        # print(f"quad: {quad_form!r}")
        log_cholesky_det = 2 * np.sum(np.log(np.diagonal(chol)))

        lp = -0.5 * (dim * np.log(2.0 * np.pi) + log_cholesky_det +
                     quad_form)
        return lp


def mat_square(mat):
    return np.matmul(mat, np.transpose(mat))


means = np.array([[[01.0, 02.0],
                   [03.0, 04.0]],
                  [[05.0, 06.0],
                   [07.0, 08.0]]])

chol_covars = np.array([[[[0.1, 0.0],
                          [0.2, 0.3]],
                         [[0.4, 0.0],
                          [0.5, 0.6]]],
                        [[[0.7, 0.0],
                          [0.8, 0.9]],
                         [[1.0, 0.0],
                          [1.1, 1.2]]]])

num_component = 2
num_latent = 2

chol_component_covar = []
component_mean = []
component_covar = []

for i in range(num_component):
    temp_cov = np.zeros([2, 2])
    temp_mean = np.zeros([2])

    for k in range(num_latent):
        # temp1 = sl.solve_triangular(chol_covars[i, k, :, :], np.eye(2), lower=True)
        # temp_cov += sl.solve_triangular(chol_covars[i, k, :, :].T, temp1)
        chol_square = chol_covars[i, k, :, :] @ chol_covars[i, k, :, :].T
        temp_cov += sl.inv(chol_square)
        # print(f"temp_cov: {temp_cov!r}")

        temp_mean += sl.inv(chol_square) @ means[i, k, :]
        # temp_mean += sl.solve_triangular(chol_covars[i, k, :, :].T, temp2)

    temp_chol_factor = sl.cho_factor(temp_cov)
    temp_component_covar = sl.cho_solve(temp_chol_factor, np.eye(num_component))
    component_covar.append(temp_component_covar)

    component_mean.append(temp_component_covar @ temp_mean)
    chol_component_covar.append(sl.cholesky(temp_component_covar))

chol_component_covar = np.stack(chol_component_covar, 0)
component_covar = np.stack(component_covar, 0)
component_mean = np.stack(component_mean, 0)

log_normal_probs = np.zeros([2, 2])
for i in range(num_component):
    for j in range(num_component):
        if i == j:
            # Compute chol(2S) = sqrt(2)*chol(S).
            chol_covars_sum = np.sqrt(2.0) * chol_component_covar[i, ...]
        else:
            covars_sum = component_covar[i, ...] + component_covar[j, ...]
            chol_covars_sum = sl.cholesky(covars_sum)

        chol_normal = CholNormal(component_mean[i, :], chol_covars_sum)
        # print(chol_normal.log_prob(means[j, :]))
        # print(normal.log_prob(means[j, k, :]))
        log_normal_probs[i][j] += chol_normal.log_prob(component_mean[j, :])

print(log_normal_probs)
entropy = 0.0
weights = np.array([0.7, 0.3])

for i in range(2):
    weighted_log_probs = np.array([0.0, 0.0])
    for j in range(2):
        if i <= j:
            weighted_log_probs[j] = np.log(weights[j]) + log_normal_probs[i][j]
        else:
            weighted_log_probs[j] = np.log(weights[j]) + log_normal_probs[j][i]

    entropy -= weights[i] * scipy.special.logsumexp(np.stack(weighted_log_probs))
    # print(entropy)

print(entropy)

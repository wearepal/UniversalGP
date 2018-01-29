# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 18:56:08 2018

@author: zc223
"""

import numpy as np
import tensorflow as tf

from . import cov
from . import mean
from . import lik
from . import inf
from . import util


class GaussianProcess:
    """
    The class is the main Gaussian Process model.

    Parameters
    ----------
    cov_func : prior covariance function (kernel).
    mean_func : prior mean function.
    lik_func : likelihood function p(y|f).
    inf_func : function specifying the inference method.

    """
    def __init__(self,
                 inducing_inputs,
                 num_components=1,
                 inf_func=inf.ExactInference,
                 cov_func=cov.SquaredExponential,
                 # mean_func=mean.ZeroOffset(),
                 lik_func=lik.LikelihoodGaussian):

        self.cov = cov_func
        # self.mean = mean_func
        self.inf = inf_func
        self.lik = lik_func

        # Repeat the inducing inputs for all latent processes if we haven't been given individually
        # specified inputs per process.
        if inducing_inputs.ndim == 2:
            inducing_inputs = np.tile(inducing_inputs[np.newaxis, :, :], [len(self.cov), 1, 1])

        # Initialize all model dimension constants.
        self.num_components = num_components
        self.num_latent = len(self.cov)
        self.num_inducing = inducing_inputs.shape[1]
        self.input_dim = inducing_inputs.shape[2]

        # Define all parameters that get optimized directly in raw form. Some parameters get
        # transformed internally to maintain certain pre-conditions.

        self.raw_weights = tf.Variable(tf.zeros([self.num_components]))
        self.raw_means = tf.Variable(tf.zeros([self.num_components, self.num_latent,
                                               self.num_inducing]))
        if self.diag_post:
            self.raw_covars = tf.Variable(tf.ones([self.num_components, self.num_latent,
                                                   self.num_inducing]))
        else:
            init_vec = np.zeros([self.num_components, self.num_latent] +
                                util.tri_vec_shape(self.num_inducing), dtype=np.float32)
            self.raw_covars = tf.Variable(init_vec)

        self.raw_inducing_inputs = tf.Variable(inducing_inputs, dtype=tf.float32)
        self.raw_likelihood_params = self.lik.get_params()
        self.raw_kernel_params = sum([k.get_params() for k in self.cov], [])

        # Define placeholder variables for training and predicting.
        self.num_train = tf.placeholder(tf.float32, shape=[], name="num_train")
        self.train_inputs = tf.placeholder(tf.float32, shape=[None, self.input_dim],
                                           name="train_inputs")
        self.train_outputs = tf.placeholder(tf.float32, shape=[None, None],
                                            name="train_outputs")
        self.test_inputs = tf.placeholder(tf.float32, shape=[None, self.input_dim],
                                          name="test_inputs")

        # if the inference is VI, the obj_func is elbo
        # else obj_func is negative log marginal likelihood
        self.cov_chol, self.obj_func = self.inf(self.mean, self.cov, self.lik,
                                                self.train_inputs,
                                                self.train_outputs)

        self.predictions = self._build_predict(self.raw_weights,
                                               self.raw_covars,
                                               self.raw_inducing_inputs,
                                               self.cov_chol,
                                               self.test_inputs)

        # config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
        # Do all the tensorflow bookkeeping.
        self.session = tf.Session()
        self.optimizer = None
        self.train_step = None

    def fit(self, data, optimizer, var_steps=10, epochs=200, batch_size=None,
            display_step=1, test=None):
        """
        Fit the Gaussian process model to the given data.

        Parameters
        ----------
        data : subclass of datasets.DataSet
            The train inputs and outputs.
        optimizer : TensorFlow optimizer
            The optimizer to use in the fitting process.
        """

        num_train = data.num_examples

        if batch_size is None:
            batch_size = num_train

        hyper_param = self.raw_kernel_params + self.raw_likelihood_params  # hyperparameters
        hyper_param = hyper_param + [self.raw_inducing_inputs]

        if self.optimizer != optimizer:
            self.optimizer = optimizer
            if self.inf is inf.VariationalInference:
                var_param = [self.raw_means, self.raw_covars, self.raw_weights]  # variational parameters
                self.train_step = optimizer.minimize(self.obj_func, var_list=var_param + hyper_param)
            else:
                self.train_step = optimizer.minimize(self.obj_func, var_list=hyper_param)

            self.session.run(tf.global_variables_initializer())

        iter_num = 0
        while data.epochs_completed < epochs:
            var_iter = 0
            while var_iter < var_steps:
                batch = data.next_batch(batch_size)
                self.session.run(self.train_step, feed_dict={self.train_inputs: batch[0],
                                                             self.train_outputs: batch[1],
                                                             self.num_train: num_train})
                if var_iter % display_step == 0:
                    self._print_state(data, test, num_train, iter_num)
                var_iter += 1
                iter_num += 1

    def predict(self, test_inputs, batch_size=None):
        """
        Predict outputs given inputs.

        Parameters
        ----------
        test_inputs : ndarray
            Points on which we wish to make predictions. Dimensions: num_test * input_dim.
        batch_size : int
            The size of the batches we make predictions on. If batch_size is None, predict on the
            entire test set at once.

        Returns
        -------
        ndarray
            The predicted mean of the test inputs. Dimensions: num_test * output_dim.
        ndarray
            The predicted variance of the test inputs. Dimensions: num_test * output_dim.
        """

        if batch_size is None:
            num_batches = 1
        else:
            num_batches = util.ceil_divide(test_inputs.shape[0], batch_size)

        test_inputs = np.array_split(test_inputs, num_batches)
        pred_means = util.init_list(0.0, [num_batches])
        pred_vars = util.init_list(0.0, [num_batches])

        for i in range(num_batches):
            pred_means[i], pred_vars[i] = self.session.run(
                self.predictions, feed_dict={self.test_inputs: test_inputs[i]})

        return np.concatenate(pred_means, axis=0), np.concatenate(pred_vars, axis=0)

    def _build_predict(self, weights, means, covars, inducing_inputs,
                       cov_chol, test_inputs):

        if self.inf is inf.ExactInference


        kern_prods, kern_sums = self._build_interim_vals(cov_chol, inducing_inputs, test_inputs)
        pred_means = util.init_list(0.0, [self.num_components])
        pred_vars = util.init_list(0.0, [self.num_components])
        for i in range(self.num_components):
            covar_input = covars[i, :, :] if self.diag_post else covars[i, :, :, :]
            sample_means, sample_vars = self._build_sample_info(kern_prods, kern_sums,
                                                                means[i, :, :], covar_input)
            pred_means[i], pred_vars[i] = self.lik.predict(sample_means, sample_vars)

        pred_means = tf.stack(pred_means, 0)
        pred_vars = tf.stack(pred_vars, 0)

        # Compute the mean and variance of the gaussian mixture from their components.
        # weights = tf.expand_dims(tf.expand_dims(weights, 1), 1)
        weights = weights[:, tf.newaxis, tf.newaxis]
        weighted_means = tf.reduce_sum(weights * pred_means, 0)
        weighted_vars = (tf.reduce_sum(weights * (pred_means ** 2 + pred_vars), 0) -
                         tf.reduce_sum(weights * pred_means, 0) ** 2)
        return weighted_means, weighted_vars

    def _print_state(self, data, test, num_train, iter_num):
        if num_train <= 100000:
            obj_func = self.session.run(self.obj_func, feed_dict={self.train_inputs: data.X,
                                                                  self.train_outputs: data.Y,
                                                                  self.num_train: num_train})
            print(f"iter={iter_num!r} [epoch={data.epochs_completed!r}] obj_func={obj_func!r}", end=" ")

    def _build_interim_vals(self, kernel_chol, inducing_inputs, train_inputs):
        kern_prods = util.init_list(0.0, [self.num_latent])
        kern_sums = util.init_list(0.0, [self.num_latent])
        for i in range(self.num_latent):
            ind_train_kern = self.cov[i].kernel(inducing_inputs[i, :, :], train_inputs)
            # Compute A = Kxz.Kzz^(-1) = (Kzz^(-1).Kzx)^T.
            kern_prods[i] = tf.transpose(tf.cholesky_solve(kernel_chol[i, :, :], ind_train_kern))
            # We only need the diagonal components.
            kern_sums[i] = (self.cov[i].diag_kernel(train_inputs) -
                            util.diag_mul(kern_prods[i], ind_train_kern))

        kern_prods = tf.stack(kern_prods, 0)
        kern_sums = tf.stack(kern_sums, 0)
        return kern_prods, kern_sums

    def _build_samples(self, kern_prods, kern_sums, means, covars):
        sample_means, sample_vars = self._build_sample_info(kern_prods, kern_sums, means, covars)
        batch_size = tf.shape(sample_means)[0]
        return (sample_means + tf.sqrt(sample_vars) *
                tf.random_normal([self.num_samples, batch_size, self.num_latent]))

    def _build_sample_info(self, kern_prods, kern_sums, means, covars):
        sample_means = util.init_list(0.0, [self.num_latent])
        sample_vars = util.init_list(0.0, [self.num_latent])
        for i in range(self.num_latent):
            if self.diag_post:
                quad_form = util.diag_mul(kern_prods[i, :, :] * covars[i, :],
                                          tf.transpose(kern_prods[i, :, :]))
            else:
                full_covar = covars[i, :, :] @ tf.transpose(covars[i, :, :])
                quad_form = util.diag_mul(kern_prods[i, :, :] @ full_covar,
                                          tf.transpose(kern_prods[i, :, :]))
            sample_means[i] = kern_prods[i, :, :] @ means[i, :, tf.newaxis]
            sample_vars[i] = (kern_sums[i, :] + quad_form)[:, tf.newaxis]

        sample_means = tf.concat(sample_means, 1)
        sample_vars = tf.concat(sample_vars, 1)
        return sample_means, sample_vars

    def rmse(self, pred_means, test_outputs):
        num_test = len(pred_means)
        mse = util.sq_dist(pred_means, test_outputs) / num_test
        return tf.sqrt(mse)
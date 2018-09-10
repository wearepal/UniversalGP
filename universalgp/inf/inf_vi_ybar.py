"""
Fair variational inference for generic Gaussian process models
"""
import tensorflow as tf
from tensorflow import manip as tft
from tensorflow import math as tfm
import numpy as np

from .. import util
from .inf_vi import Variational

# General fairness
tf.app.flags.DEFINE_float('biased_acceptance1', 0.5, '')
tf.app.flags.DEFINE_float('biased_acceptance2', 0.5, '')
tf.app.flags.DEFINE_boolean('s_as_input', True,
                            'Whether the sensitive attribute is treated as part of the input')
tf.app.flags.DEFINE_float('p_s0', 0.5, '')
tf.app.flags.DEFINE_float('p_s1', 0.5, '')
# Demographic parity
tf.app.flags.DEFINE_float('target_rate1', 0.5, '')
tf.app.flags.DEFINE_float('target_rate2', 0.5, '')
tf.app.flags.DEFINE_boolean('probs_from_flipped', False,
                            'Whether to take the target rates from the flipping probs')
tf.app.flags.DEFINE_boolean('average_prediction', False,
                            'Whether to take the average of both sensitive attributes')
tf.app.flags.DEFINE_float('p_ybary0_or_ybary1_s0', 1.0,
                          'Determine how similar the target labels are to the true labels for s=0')
tf.app.flags.DEFINE_float('p_ybary0_or_ybary1_s1', 1.0,
                          'Determine how similar the target labels are to the true labels for s=1')
# Equalized Odds
tf.app.flags.DEFINE_float('p_ybary0_s0', 1.0, '')
tf.app.flags.DEFINE_float('p_ybary1_s0', 1.0, '')
tf.app.flags.DEFINE_float('p_ybary0_s1', 1.0, '')
tf.app.flags.DEFINE_float('p_ybary1_s1', 1.0, '')


class VariationalWithS(Variational):
    """Thin wrapper around `Variational` that can handle sensitive attributes"""
    def predict(self, test_inputs):
        if self.args['s_as_input']:
            s = test_inputs['sensitive']
            if self.args['average_prediction']:
                preds_s0 = super().predict({'input': tf.concat((test_inputs['input'],
                                                                tf.zeros_like(s)), axis=1)})
                preds_s1 = super().predict({'input': tf.concat((test_inputs['input'],
                                                                tf.ones_like(s)), axis=1)})
                return [self.args['p_s0'] * r_s0 + self.args['p_s1'] * r_s1
                        for r_s0, r_s1 in zip(preds_s0, preds_s1)]
            return super().predict({'input': tf.concat((test_inputs['input'], s), axis=1)})
        return super().predict(test_inputs)

    def _build_ell(self, weights, means, chol_covars, inducing_inputs, kernel_chol, features,
                   outputs, is_train):
        if self.args['s_as_input']:
            inputs = tf.concat((features['input'], features['sensitive']), axis=1)
        else:
            inputs = features['input']
        return super()._build_ell(weights, means, chol_covars, inducing_inputs, kernel_chol,
                                  {'input': inputs}, outputs, is_train)


class VariationalYbar(VariationalWithS):
    """Defines inference for simple fair Variational Inference"""
    def _build_ell(self, weights, means, chol_covars, inducing_inputs, kernel_chol, features,
                   outputs, is_train):
        """Construct the Expected Log Likelihood

        Args:
            weights: (num_components,)
            means: shape: (num_components, num_latents, num_inducing)
            chol_covars: shape: (num_components, num_latents, num_inducing[, num_inducing])
            inducing_inputs: (num_latents, num_inducing, input_dim)
            kernel_chol: (num_latents, num_inducing, num_inducing)
            inputs: (batch_size, input_dim)
            outputs: (batch_size, num_latents)
            is_train: True if we're training, False otherwise
        Returns:
            Expected log likelihood as scalar
        """
        if self.args['s_as_input']:
            inputs = tf.concat((features['input'], features['sensitive']), axis=1)
        else:
            inputs = features['input']

        kern_prods, kern_sums = self._build_interim_vals(kernel_chol, inducing_inputs, inputs)
        # shape of `latent_samples`: (num_components, num_samples, batch_size, num_latents)
        latent_samples = self._build_samples(kern_prods, kern_sums, means, chol_covars)
        if is_train:
            sens_attr = tf.cast(tf.squeeze(features['sensitive'], -1), dtype=tf.int32)
            out_int = tf.cast(tf.squeeze(outputs, -1), dtype=tf.int32)
            log_lik0 = self.lik.log_cond_prob(tf.zeros_like(outputs), latent_samples)
            log_lik1 = self.lik.log_cond_prob(tf.ones_like(outputs), latent_samples)
            log_lik = tf.stack((log_lik0, log_lik1), axis=-1)
            debias = self._debiasing_parameters()
            # `debias` has the shape (y, s, y'). we stack output and sensitive to (batch_size, 2)
            # then we use the last 2 values of that as indices for `debias`
            # shape of debias_per_example: (batch_size, output_dim, 2)
            debias_per_example = tft.gather_nd(debias, tf.stack((out_int, sens_attr), axis=-1))
            weighted_lik = debias_per_example * tf.exp(log_lik)
            log_cond_prob = tfm.log(tf.reduce_sum(weighted_lik, axis=-1))
        else:
            log_cond_prob = self.lik.log_cond_prob(outputs, latent_samples)
        ell_by_component = tf.reduce_sum(log_cond_prob, axis=[1, 2])

        # weighted sum of the components
        ell = util.mul_sum(weights, ell_by_component)
        return ell / self.args['num_samples']

    def _debiasing_parameters(self):
        return debiasing_params_target_rate(self.args)


class VariationalYbarEqOdds(VariationalYbar):
    """
    Defines inference for Variational Inference that enforces Equalized Odds
    """
    def _debiasing_parameters(self):
        return debiasing_params_target_tpr(self.args)


def compute_label_posterior(positive_value, positive_prior, label_evidence=None):
    """Return label posterior from positive likelihood P(y'=1|y,s) and positive prior P(y=1|s)

    Args:
        positive_value: P(y'=1|y,s), shape (y, s)
        label_prior: P(y|s)
    Returns:
        Label posterior, shape (y, s, y')
    """
    # compute the prior
    # P(y=0|s)
    negative_prior = 1 - positive_prior
    # P(y|s) shape: (y, s, 1)
    label_prior = np.stack([negative_prior, positive_prior], axis=0)[..., np.newaxis]

    # compute the likelihood
    # P(y'|y,s) shape: (y, s, y')
    label_likelihood = np.stack([1 - positive_value, positive_value], axis=-1)

    # compute joint and evidence
    # P(y',y|s) shape: (y, s, y')
    joint = label_likelihood * label_prior
    # P(y'|s) shape: (s, y')
    if label_evidence is None:
        label_evidence = np.sum(joint, axis=0)

    # compute posterior
    # P(y|y',s) shape: (y, s, y')
    label_posterior = joint / label_evidence
    return tf.constant(label_posterior, dtype=tf.float32)


def debiasing_params_target_rate(args):
    """Debiasing parameters for implementing target acceptance rates

    Args:
        args: dictionary with parameters
    Returns:
        P(y|y',s) with shape (y, s, y')
    """
    if args['probs_from_flipped']:
        biased_acceptance1 = 0.5 * (1 - args['reject_flip_probability'])
        biased_acceptance2 = 0.5 * (1 + args['accept_flip_probability'])
    else:
        biased_acceptance1 = args['biased_acceptance1']
        biased_acceptance2 = args['biased_acceptance2']
    # P(y'=1|s)
    target_acceptance = np.array([args['target_rate1'], args['target_rate2']])
    # P(y=1|s)
    positive_prior = np.array([biased_acceptance1, biased_acceptance2])
    # P(y'=1|y,s) shape: (y, s)
    positive_value = positive_label_likelihood(args, positive_prior, target_acceptance)
    # P(y'|s) shape: (s, y')
    label_evidence = np.stack([1 - target_acceptance, target_acceptance], axis=-1)
    return compute_label_posterior(positive_value, positive_prior, label_evidence)


def positive_label_likelihood(args, biased_acceptance, target_acceptance):
    """Compute the label likelihood (for positive labels)

    Args:
        biased_acceptance: P(y=1|s)
        target_acceptance: P(y'=1|s)
    Returns:
        P(y'=1|y,s) with shape (y, s)
    """
    positive_lik = []
    for i, (target, biased) in enumerate(zip(target_acceptance, biased_acceptance)):
        if target > biased:
            # P(y'=1|y=1)
            p_ybary1 = args[f"p_ybary0_or_ybary1_s{i}"]
            # P(y'=1|y=0) = (P(y'=1) - P(y'=1|y=1)P(y=1))/P(y=0)
            p_ybar1_y0 = (target - p_ybary1 * biased) / (1 - biased)
        else:
            p_ybar1_y0 = 1 - args[f"p_ybary0_or_ybary1_s{i}"]
            # P(y'=1|y=0) = (P(y'=1) - P(y'=1|y=0)P(y=0))/P(y=1)
            p_ybary1 = (target - p_ybar1_y0 * (1 - biased)) / biased
        positive_lik.append([p_ybar1_y0, p_ybary1])
    positive_lik_arr = np.array(positive_lik)  # shape: (s, y)
    return np.transpose(positive_lik_arr)  # shape: (y, s)


def debiasing_params_target_tpr(args):
    """Debiasing parameters for targeting TPRs and TNRs

    Args:
        args: dictionary with parameters
    Returns:
        P(y|y',s) with shape (y, s, y')
    """
    # P(y=1|s)
    positive_prior = np.array([args['biased_acceptance1'], args['biased_acceptance2']])
    # P(y'=1|y=1,s)
    positive_predictive_value = np.array([args['p_ybary1_s0'], args['p_ybary1_s1']])
    # P(y'=0|y=0,s)
    negative_predictive_value = np.array([args['p_ybary0_s0'], args['p_ybary0_s1']])
    # P(y'=1|y=0,s)
    false_omission_rate = 1 - negative_predictive_value
    # P(y'=1|y,s) shape: (y, s)
    positive_value = np.stack([false_omission_rate, positive_predictive_value], axis=0)
    return compute_label_posterior(positive_value, positive_prior)

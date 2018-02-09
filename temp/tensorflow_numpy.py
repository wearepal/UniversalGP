import tensorflow as tf
import tensorflow.contrib.eager as tfe

tfe.enable_eager_execution()

import universalgp

lik = universalgp.lik.LikelihoodGaussian()
cov = [universalgp.cov.SquaredExponential(1)]
inf = universalgp.inf.inf_vi.Variational([cov, cov], lik, num_components=2)

weights = tf.constant([0.7, 0.3])

means = tf.constant([[[01.0, 02.0],
                   [03.0, 04.0]],
                  [[05.0, 06.0],
                   [07.0, 08.0]]])

chol_covars = tf.constant([[[[0.1, 0.0],
                          [0.2, 0.3]],
                         [[0.4, 0.0],
                          [0.5, 0.6]]],
                        [[[0.7, 0.0],
                          [0.8, 0.9]],
                         [[1.0, 0.0],
                          [1.1, 1.2]]]])
print(weights.shape, means.shape, chol_covars.shape)

ent = inf._build_entropy(weights, means, chol_covars)
print(ent)

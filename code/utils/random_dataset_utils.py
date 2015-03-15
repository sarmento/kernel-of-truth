## 
## Set of utility functions for generating datasets
## Not intended to be high-performance: just easy to use.
##

import numpy as np

def generate_linear_relation_dataset(linear_weights_vector, n_samples):
  n_X_dimensions = linear_weights_vector.shape[0]  
  ## this is for the intercept component (all ones)
  X_0 = np.ones([n_samples, 1])
  ## Genearate random values for the stochastic components of X
  X_stochastic = np.random.uniform(-1, 1, [n_samples, n_X_dimensions - 1])
  X = np.hstack((X_0, X_stochastic))
  
  Y = X.dot(linear_weights_vector)
  return [X,Y]


def add_gaussian_noise(Y, mu, sigma, ratio_of_points_affected):
  for ii in range(Y.shape[0]):
    ## should we try to add noise?
    if np.random.uniform(0,1) > ratio_of_points_affected:
      continue
    Y[ii] += np.random.normal(mu, sigma)
  return Y
 

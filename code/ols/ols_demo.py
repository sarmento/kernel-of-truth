import numpy as np
import sys
sys.path.append('../utils')

## our own packages
import ols as ols
import random_dataset_utils as rdu
import plot_utils as pu

n_runs = 10000
n_samples = 500

## Let's define some true values for the w vector, which
## later we will try to estimate from the noisy data.
## Note that w_true is a vertical vector

w_true = np.array([[1], \
                   [2]])

(X,Y_true) = rdu.generate_linear_relation_dataset(w_true, n_samples)

w_hat_list = []  ## This is where we are going to store all the estimatec w_hat's
runs_so_far = 0
while (runs_so_far < n_runs):

  ## Add gaussian noise to all points
  Y = rdu.add_gaussian_noise(Y_true, 0, 1, 1)
  w_hat = ols.solve_ols(X,Y)  ## w_hat is a vertical vector
  w_hat_list.append(w_hat)
  runs_so_far += 1
  if runs_so_far % 1000 == 0:
    print "Executed ", runs_so_far, "runs so far..."

## Now let's plot the distributions of w_0 and w_1

w_matrix = pu.list_of_vertical_vectors_to_matrix(w_hat_list, w_true.shape[0])
print "Plotting distribution of w_0..."
pu.plot_histogram(w_matrix[0,:], \
                  "Distribution of values of $\hat{w}_0$ (" + str(n_runs) + " runs). True $w_0$ = 1", \
                  'green', \
                  '$w_0$', \
                  'nr. runs', \
                   1, \
                   "")
print "Plotting distribution of w_1..."
pu.plot_histogram(w_matrix[1,:], \
                  "Distribution of values of $\hat{w}_1$ ("  + str(n_runs) + " runs). True $w_1$ = 2", \
                  'red', \
                  '$w_1$', \
                  'nr. runs', \
                   1, \
                   "")

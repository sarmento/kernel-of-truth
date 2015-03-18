import numpy as np
import sys
sys.path.append('../utils')
sys.path.append('../ols')

## our own packages
import ols as ols
import random_dataset_utils as rdu
import plot_utils as pu

n_runs = 10000
n_samples = 500
p_outliers = 0.10

## Let's define some true values for the w vector, which
## later we will try to estimate from the noisy data.
## Note that w_true is a vertical vector

w_true = np.array([[1], \
                   [2]])

(X,Y_true) = rdu.generate_linear_relation_dataset(w_true, n_samples)

w_hat_list = []           ## This is where we are going to store all the estimated w_hat's
w_hat_outliers_list = []  ## and this is for the case of outliers
runs_so_far = 0
while (runs_so_far < n_runs):

  ## Add gaussian noise to all points
  Y = rdu.add_gaussian_noise(Y_true, 0, 1, 1)
  ## Add 1% of outliers (gaussian noise with much higher variance!)
  Y_outliers = rdu.add_gaussian_noise(Y, 0, 5, p_outliers)
  w_hat = ols.solve_ols(X,Y)  ## w_hat is a vertical vector
  w_hat_list.append(w_hat)
  w_hat_outliers = ols.solve_ols(X,Y_outliers)  ## w_hat is a vertical vector
  w_hat_outliers_list.append(w_hat_outliers)
  runs_so_far += 1
  if runs_so_far % 1000 == 0:
    print "Executed ", runs_so_far, "runs so far..."

## Now let's plot the distributions of w_0 and w_1
percent_outliers = str(int(100 * p_outliers))
w_matrix = pu.list_of_vertical_vectors_to_matrix(w_hat_list, w_true.shape[0])
w_matrix_outliers = pu.list_of_vertical_vectors_to_matrix(w_hat_outliers_list, w_true.shape[0])

print "Plotting distribution of w_0..."
histogram_plot = pu.HistogramPlot("Distribution of values of $\hat{w}_0$ (" + str(n_runs) + " runs / " + percent_outliers + "% outliers). True $w_0$ = 1", \
                                  '$w_0$', \
                                  'nr. runs')

histogram_plot.add_data_series(w_matrix[0,:], '#009900')
histogram_plot.add_data_series(w_matrix_outliers[0,:], '#00DD00')
histogram_plot.show()

print "Plotting distribution of w_1..."
histogram_plot = pu.HistogramPlot("Distribution of values of $\hat{w}_1$ (" + str(n_runs) + " runs / " + percent_outliers + "% outliers). True $w_1$ = 2", \
                                  '$w_1$', \
                                  'nr. runs')

histogram_plot.add_data_series(w_matrix[1,:], '#990000')
histogram_plot.add_data_series(w_matrix_outliers[1,:], '#DD0000')
histogram_plot.show()


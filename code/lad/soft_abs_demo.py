import numpy as np
import sys
sys.path.append('../utils')
import lad as lad
import random_dataset_utils as rdu


n_x_dimensions = 1
n_y_dimensions = 1
n_runs_left = 10000
n_samples = 500

## Let's define some true values for the w vector, which
## later we will try to estimate from the noisy data.
linear_weigths = np.array([[1],[2]])

(X,Y) = rdu.generate_linear_relation_dataset(linear_weigths, n_samples)

## Add a moderate amount of gaussion noise to all points
Y = rdu.add_gaussian_noise(Y, 0, 1, 1)

w_hat = lad.solve_lad_soft(X,Y, 1000000)

print w_hat

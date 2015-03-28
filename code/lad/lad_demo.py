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

## Now add 5% of outliers just to make the problem harder

w_hat = lad.solve_lad(X,Y)
print w_hat




## So now we can generate the corresponding Y's given the X that 
## we've generated before and the true weights we defined as our 
## model. We will add noise to the observed Y so that we can then 
## see what weight estimates we actually get.
## For now we assume gaussian noise, m = 0, sigma^2 = 1
##w_hat_vec = np.empty([n_x_dimensions + 1, 0])
##while (n_runs_left > 0):
## 
##  ##noise = np.random.normal(0, 1, [n_samples, n_y_dimensions])
##  noise = np.random.uniform(-1, 1, [n_samples, n_y_dimensions])
##  Y = X.dot(w) + noise
##  
##  ## Now given our observed X and Y, let's solve the OLS to
##  ## estimate w_hat, which is given by:
##  ##
##  ## w_hat = (X^T X)^-{1} X^T Y
##  ##
##
##  XtX = X.transpose().dot(X)
##  XtY = X.transpose().dot(Y)
##  w_hat = np.linalg.inv(XtX).dot(XtY)
##  ## w_hat is a vector of shape [n_x_dimensions + 1, 1]
##  w_hat_vec = np.hstack([w_hat_vec, w_hat])
##
##  n_runs_left -= 1


## Now let's plot the distributions of w_0 and w_1
##print "Here are the w_0"
##print w_hat_vec[0,:]
##
##print "Here are the w_1"
##print w_hat_vec[1,:]

##plt.hist(w_hat_vec[0,:], 100, normed=0, facecolor='green', alpha=0.5)
##plt.title(r'Histogram of values of $w_0$ (100000 runs). True $w_0$ = 1')
##plt.xlabel('$w_0$ / intercept')
##plt.ylabel('nr. runs')
##plt.show()

##plt.hist(w_hat_vec[1,:], 100, normed=0, facecolor='red', alpha=0.5)
##plt.title(r'Histogram of values of $w_2$ (100000 runs). True $w_1$ = 2')
##plt.xlabel('$w_1$')
##plt.ylabel('nr. runs')
##plt.show()




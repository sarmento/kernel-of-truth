import numpy as np
from cvxopt.modeling import variable, op, sum, matrix, solvers

## We want to minimize
##
## min Sum_i(ald_i) = 
##     Sum_i | y_i - X.dot(w_hat) |
##
##
## Alternatively, we want to minimize the sum of
## the majorants of each ald_i, Ald_i
##
## min Sum_i(Ald_i) 
## 
## s.t. Ald_i >=    y_i - X.dot(w_hat) 
##      Ald_i >= - (y_i - X.dot(w_hat))
##
##
## Nota that these constraints force the majorants Ald_i to be 
## equal to ald_i after minimization but we get rid of the | |
## This has now become a standard linear programing problem.

## Using cvxopt notation
##
## 
## min:   1 ald_1 + 1 ald_2 ... 1 ald_3    c = [ 1, 1, 1, 1,.. 1]
##
## s.t:
##        
##  it gets messy... but cvxopt allows us to model this directly
##
##
##

def solve_lad(X,Y):
  Y_cvx = matrix(Y)
  X_cvx = matrix(X)
  w_hat = variable(X.shape[1])
  solvers.options['show_progress'] = False
  op(sum(abs(Y_cvx - X_cvx*w_hat))).solve()
  return w_hat.value

def solve_lad_soft(X,Y, n_iter):
  w_hat = np.random.rand(X.shape[1],1) - 0.5
  while n_iter > 0:
    D = Y - X.dot(w_hat)
    COSH2 = np.cosh(D) * np.cosh(D)
    D1 = np.tanh(D) + D / COSH2
    G = X.transpose().dot(D1)
    w_hat = w_hat + 0.0001 * G
    n_iter -= 1
  return w_hat

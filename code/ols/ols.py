import numpy as np


## Given our observed X and Y, let's solve the OLS to
## estimate w_hat, which is given by:
##
## w_hat = (X^T X)^-{1} X^T Y
##

def solve_ols(X,Y):
  XtX = X.transpose().dot(X)
  XtY = X.transpose().dot(Y)
  w_hat = np.linalg.inv(XtX).dot(XtY)
  return w_hat


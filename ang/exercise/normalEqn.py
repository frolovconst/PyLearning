#import numpy as np
from numpy.linalg import inv
def normalEqn(X, y):
    theta_nEqn = (inv((X.T).dot(X)).dot(X.T).dot(y))
    return theta_nEqn

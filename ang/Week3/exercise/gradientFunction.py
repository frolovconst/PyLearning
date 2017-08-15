import sigmoidFunction
import numpy as np
def gradFunction(X, y, theta):
    m = y.size
    grad = np.zeros(theta.size)
    grad = (X.T.dot(sigmoidFunction.sigmoidFunction(X, theta) - y))/m
    return  grad

import sigmoidFunction
import numpy as np
def costFunction(X, y, theta):
    m = y.size
    J = -(y.T.dot(np.log(sigmoidFunction.sigmoidFunction(X, theta))) + (1-y.T).dot(np.log(1-sigmoidFunction.sigmoidFunction(X, theta))))/m
    return J

import numpy as np
import costFunctionJ
def gradientDescent(X, y, theta, alpha, num_iters):
    m = X.shape[0]
    temp_theta = theta
    J_history = np.zeros(num_iters)
    for i in range (num_iters):
        J_history[i] = costFunctionJ.costFunctionJ(X, y, theta)
        for j in range(theta.shape[0]):
            temp_theta[j] = theta[j] - alpha / m * ((X.dot(theta) - y) * X[:,j].reshape(m,1)).sum()
        theta = temp_theta
    return theta, J_history

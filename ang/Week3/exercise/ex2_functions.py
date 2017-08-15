import numpy as np

def sigmoidFunction(x, theta):
    return 1/(1+np.e**-x.dot(theta.reshape(theta.size, 1)))

def costFunction(X, y, theta):
	m = y.size
	J = -(y*np.log(sigmoidFunction(X, theta)) + (1-y)*np.log(1e-15 + 1-sigmoidFunction(X, theta))).sum()/m
	return J

def gradFunction(X, y, theta):
	m = y.size
	grad = X.T.dot((sigmoidFunction(X, theta) - y))/m
	grad1 = 0
	grad2 = 0
#	return np.asarray((grad0, grad1, grad3))
	return grad

def predict(X, theta):
	m = X.shape[0]
	h = np.zeros(m).reshape(m,1)
	h[np.where(sigmoidFunction(X, theta).flatten()>=.5), 0] = 1
	return h

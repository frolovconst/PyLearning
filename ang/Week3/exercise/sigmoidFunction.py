import numpy as np
def sigmoidFunction(x, theta):
	return 1/(1+np.e**-x.dot(theta))

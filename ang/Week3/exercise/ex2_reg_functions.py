import matplotlib.pyplot as plt
import numpy as np
def plotData(X,y):
        pos = np.where(y[:,0]>0)
        neg = np.where(y[:,0]<1)
        pos_plot = plt.plot(X[pos][:,0], X[pos][:,1], marker='x', markeredgecolor='black', linestyle='', label='Pass')
        neg_plot = plt.plot(X[neg][:,0], X[neg][:,1], marker='o', markerfacecolor='yellow', linestyle='', label='Fail')
        plt.legend(loc=1)
        plt.xlabel('Microchip test 1')
        plt.ylabel('Microchip test 2')
        plt.show()

def mapFeature(x1, x2):
	degree = 6
	m = x1.size
	x1 = x1.reshape(m,1)
	x2 = x2.reshape(m,1)
	out = np.ones(m).reshape(m,1)
	for i in range(1, degree+1):
		#print('i=%d' %i)
		for j in range(i+1):
			out = np.append(out, x1**(i-j)*x2**(j), axis = 1)
	return out

def sigmoidFunction(X, theta):
	tht = theta.reshape(theta.size, 1)
	return 1/(1+np.e**-(X.dot(tht)))

def costFunctionReg(X, y, theta, lambda_reg):
	m = y.size
	return (lambda_reg/2*(theta[1:]**2).sum()-(y*np.log(sigmoidFunction(X, theta)) + (1-y)*np.log(1e-15 + 1-sigmoidFunction(X, theta))).sum())/m

def gradFunction(X, y, theta, lambda_reg):
	m = y.size
	n = X.shape[1]
	return (X.T.dot(sigmoidFunction(X,theta) - y)+ lambda_reg*np.append(np.zeros(1), theta.flatten()[1:], axis=0).reshape(n,1)
)/m
 

def predict(X, theta):
        m = X.shape[0]
        h = np.zeros(m).reshape(m,1)
        h[np.where(sigmoidFunction(X, theta).flatten()>=.5), 0] = 1
        return h


	
def plotDecisionBoundary(X, y, theta):
	u = np.linspace(-1, 1.5, 50)
	v = np.linspace(-1, 1.5, 50)
	size = u.size
	z = np.zeros(size **2).reshape(size, size)
	for i in range(size):
		for j in range(size):
			z[i,j] = (mapFeature(u[i], v[j]).dot(theta)).sum()
	z = z.T
	
	pos = np.where(y[:,0]>0)
	neg = np.where(y[:,0]<1)
	pos_plot = plt.plot(X[pos][:,0], X[pos][:,1], marker='x', markeredgecolor='black', linestyle='', label='Pass')
	neg_plot = plt.plot(X[neg][:,0], X[neg][:,1], marker='o', markerfacecolor='yellow', linestyle='', label='Fail')
	plt.legend(loc=1)
	plt.xlabel('Microchip test 1')
	plt.ylabel('Microchip test 2')
	cplot = plt.contour(u, v, z, levels=np.array([0]))
	plt.show()
	

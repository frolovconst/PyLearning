import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import optimize



def getDatumImg(row):
#	"""
#	Function that is handed a single np array with shape 1x400,
#	crates an image object from it, and returns it
#	"""
	width, height = 20, 20
	square = row[1:].reshape(width,height)
	return square.T
    
def displayData(X, y, indices_to_display = None):
#	"""
#	Function that picks 100 random rows from X, creates a 20x20 image from each,
#	then stitches them together into a 10x10 grid of images, and shows it.
#	"""
	width, height = 20, 20
	nrows, ncols = 10, 10
	if not indices_to_display.all():
		indices_to_display = random.sample(range(X.shape[0]), nrows*ncols)
	    
	big_picture = np.zeros((height*nrows,width*ncols))
	
	irow, icol = 0, 0
	for idx in indices_to_display:
		if icol == ncols:
			irow += 1
			icol  = 0
		iimg = getDatumImg(X[idx])
		big_picture[irow*height:irow*height+iimg.shape[0],icol*width:icol*width+iimg.shape[1]] = iimg
		icol += 1
	fig = plt.figure(figsize=(6,6))
	img = scipy.misc.toimage( big_picture )
	plt.imshow(img,cmap = cm.Greys_r)
	plt.show()

def sigmoid(X, theta):
	tht = theta.reshape(theta.size, 1)
	return 1/(1+np.e**-(X.dot(tht)))

def sigmoidMatrix(X, thetas):
	return 1/(1+np.e**-(X.dot(thetas)))

def lrCostFunction(X, y, theta, lmbd):
	m = y.size
	fst_part = (y-1).T.dot(np.log(1e-10 + 1-sigmoid(X, theta)))
	scnd_part = y.T.dot(np.log(sigmoid(X,theta)))
	thrd_part = lmbd * np.insert(theta[1:]**2, 0, 0, axis=0).sum()/2
	return ((fst_part - scnd_part + thrd_part)/m).sum() 

def lrGradientFunction(X, y, theta, lmbd):
	m = y.size
	tht = theta.reshape(theta.size,1)
	grad = (X.T.dot((sigmoid(X, tht) - y)) + lmbd*np.insert(tht[1:], 0, 0, axis=0))/m
	return grad

def oneVsAll(X, y, lmbd, num_labels):
	m = y.size
	n = X.shape[1]
	X_d = np.insert(X, 0, 1, axis=1)
	initial_theta = np.zeros(n+1, 1)
	result = optimize.fmin_cg

def predict(X, theta, label):
	m = X.shape[0]
	h = np.zeros(m).reshape(m, 1)
	tht = theta.reshape(theta.size, 1)
	h = (X.dot(tht) >= 0) * label
	return h

def predictOneVsAll(X, thetas):
	thetas_t = thetas.T
	thts = np.insert(thetas_t, thetas_t.shape[1], thetas_t[:,0], axis=1)
	X_p = np.argmax(sigmoidMatrix(X, thts[:,1:]), axis=1) + 1
	X_p = X_p.reshape(X_p.size, 1)
	return X_p

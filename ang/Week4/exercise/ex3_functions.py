import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm



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

def lrCostFunction(X, y, theta, lmbd):
	m = y.size
	fst_part = (y-1).T.dot(np.log(1-sigmoid(X, theta)))
	print(fst_part)
	scnd_part = y.T.dot(np.log(sigmoid(X,theta)))
	print(scnd_part)
	return (fst_part - scnd_part)/m 
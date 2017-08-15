import matplotlib.pyplot as plt
import numpy as np
def plotData(X,y):
	pos = np.where(y[:,0]>0)
	neg = np.where(y[:,0]<1)
	pos_plot = plt.plot(X[pos][:,0], X[pos][:,1], marker='x', markeredgecolor='black', linestyle='', label='Admitted')
	neg_plot = plt.plot(X[neg][:,0], X[neg][:,1], marker='o', markerfacecolor='yellow', linestyle='', label='Not admitted')
	plt.legend(loc=1)
	plt.xlabel('Exam 1 score')
	plt.ylabel('Exam 2 score')
	plt.show()

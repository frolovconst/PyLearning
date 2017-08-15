import matplotlib.pyplot as plt
import numpy as np
def plotData(X,y,theta):
	pos = np.where(y[:,0]>0)
	neg = np.where(y[:,0]<1)
	pos_plot = plt.plot(X[pos][:,1], X[pos][:,2], marker='x', markeredgecolor='black', linestyle='', label='Admitted')
	neg_plot = plt.plot(X[neg][:,1], X[neg][:,2], marker='o', markerfacecolor='yellow', linestyle='', label='Not admitted')
	plt_x = np.array([min(X[:,1])-2, max(X[:,1])+2])
	plt_y = -(theta[0] + theta[1] * plt_x)/theta[2]
	dec_bndr = plt.plot(plt_x, plt_y)
	plt.legend(loc=1)
	plt.xlabel('Exam 1 score')
	plt.ylabel('Exam 2 score')
	plt.show()

import numpy as np

def squareThisNumber(x):
    return x**2, x**3

def magic_square(N):

    magic_square = np.zeros((N,N), dtype=int)

    n = 1
    i, j = 0, N//2

    while n <= N**2:
        magic_square[i, j] = n
        n += 1
        newi, newj = (i-1) % N, (j+1)% N
        if magic_square[newi, newj]:
            i += 1
        else:
            i, j = newi, newj

    return magic_square

def costFunctionJ(X, y, theta):
	m = X.shape[0]
	predictions = X.dot(theta)
	sqrErrors = predictions - y
	return sum(sqrErrors**2/2/m)

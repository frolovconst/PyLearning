def costFunctionJ(X, y, theta):
        m = X.shape[0]
        predictions = X.dot(theta)
        sqrErrors = predictions - y
        return sum(sqrErrors**2/2/m)

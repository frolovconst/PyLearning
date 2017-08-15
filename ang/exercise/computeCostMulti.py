def costFunctionJ(X, y, theta):
        m = X.shape[0]
        predictions = X.dot(theta)
        sqrErrors = predictions - y
        return ((sqrErrors.T).dot(sqrErrors))/2/m

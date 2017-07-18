import numpy as np
def featureNormalize(X):
    X_norm = X
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X_norm - mu) / sigma
    return X_norm, mu, sigma

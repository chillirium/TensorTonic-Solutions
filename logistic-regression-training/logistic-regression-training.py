import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X_train = np.asarray(X, dtype=float)
    y_train = np.asarray(y, dtype=float)

    w, b = np.zeros_like(X_train[0]), np.float32(0.)
    dN = 1. / len(y)

    for _ in range(steps):
        logits = X_train @ w + b
        p = _sigmoid(logits)

        err = p - y_train
    
        w = w - lr * (X_train.T @ err) * dN
        b = b - lr * err.sum() * dN

    return w, b
import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    X_np = np.asarray(X, dtype=float)
    if X_np.ndim != 2:
        return None

    N, D = X_np.shape

    if N < 2 or D == 0:
        return None

    X_centered = X_np - X_np.mean(axis=0)
    S = X_centered.T @ X_centered / (N - 1)
    
    return S
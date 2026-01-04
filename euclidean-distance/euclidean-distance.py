import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    x_np = np.asarray(x, dtype=float)
    y_np = np.asarray(y, dtype=float)
    return np.hypot.reduce(x_np - y_np)
import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    x_np = np.asarray(x, dtype=float)
    res = np.empty_like(x_np)
    pos = x_np >= 0

    res[pos] = x_np[pos] 
    res[~pos] = alpha * x_np[~pos]
    return res
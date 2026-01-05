import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    x_np = np.asarray(x, dtype=float)
    if x_np.ndim == 0:
        x_np = x_np.reshape(1)
    return np.maximum(0., x_np)
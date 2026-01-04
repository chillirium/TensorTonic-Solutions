import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    x_max = x.max()
    x_exp = np.exp(x - x_max)
    return x_exp / np.sum(x_exp, axis=-1, keepdims=1)
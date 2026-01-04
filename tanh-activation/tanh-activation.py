import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        x = x.reshape(1)
    return np.tanh(x)
    #return np.array(np.tanh(x), dtype=float)#не проходит)
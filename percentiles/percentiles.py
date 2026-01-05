import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """
    x_np, q_np = map(lambda z: np.asarray(z, dtype=float), (x, q))
    
    return np.percentile(x_np, q_np, method='linear')
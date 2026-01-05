import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Returns: h_t of shape (H,)
    """
    x_t_np, h_prev_np, Wx_np, Wh_np, b_np = map(lambda z: np.array(z, dtype=float), (x_t, h_prev, Wx, Wh, b))
    pre_act = x_t_np @ Wx_np + h_prev_np @ Wh_np + b_np
    h_t = np.tanh(pre_act).reshape(h_prev_np.shape)
    return h_t

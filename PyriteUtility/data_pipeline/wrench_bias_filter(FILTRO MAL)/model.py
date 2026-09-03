"""
Wrench-bias MLP: maps EE position (x, y, z) -> wrench bias (fx,fy,fz,tx,ty,tz).

The network is deliberately tiny (3 -> 32 -> 16 -> 6, ~750 params). The bias is a
smooth, low-frequency function of position, so a small MLP both fits well and
avoids overfitting the few hundred calibration poses.

Normalisation (input standardisation and output standardisation) is kept OUTSIDE
the nn.Module so the exported weights are exactly the three Linear layers. The
C++ side applies the same normalisation around the raw matrix multiplies.
"""

import torch
import torch.nn as nn

INPUT_DIM  = 14  # q[7] + tau_J[7]  (set dynamically in train.py from data shape)
HIDDEN1    = 128
HIDDEN2    = 64
OUTPUT_DIM = 6

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "elu":  nn.ELU,
}


class WrenchBiasMLP(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, hidden1: int = HIDDEN1,
                 hidden2: int = HIDDEN2, dropout: float = 0.0, activation: str = "relu"):
        super().__init__()
        act = _ACTIVATIONS.get(activation, nn.ReLU)
        # Sequential layout matters for export: indices 0/3/6 are the Linear layers.
        # Dropout is only active during training (model.train()); eval/C++ inference unaffected.
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),  # 0
            act(),                          # 1
            nn.Dropout(dropout),            # 2
            nn.Linear(hidden1, hidden2),    # 3
            act(),                          # 4
            nn.Dropout(dropout),            # 5
            nn.Linear(hidden2, OUTPUT_DIM), # 6
        )

    def forward(self, x):
        return self.net(x)

    def linear_layers(self):
        """Return the three nn.Linear modules in forward order."""
        return [self.net[0], self.net[3], self.net[6]]


def numpy_forward(weights: dict, xyz):
    """
    Reference forward pass in pure NumPy, used to verify that the exported
    weights reproduce the trained PyTorch model bit-for-bit before deployment.
    This mirrors exactly what the C++ Eigen implementation will compute.
    Dropout is not applied here (eval mode = dropout disabled).

    `weights` is the dict produced by export.py (W1,b1,W2,b2,W3,b3 + norm stats).
    `xyz` is (N_features,) or (B, N_features) — [q, tau_J] or just q.
    """
    import numpy as np
    from scipy.special import erf

    act_id = int(weights.get("activation", np.array(0)))

    def _act(x):
        if act_id == 1:  # gelu
            return x * 0.5 * (1.0 + erf(x / np.sqrt(2.0)))
        if act_id == 2:  # tanh
            return np.tanh(x)
        if act_id == 3:  # elu
            return np.where(x >= 0, x, np.expm1(x))
        return np.maximum(x, 0.0)  # relu (default)

    x = np.atleast_2d(np.asarray(xyz, dtype=np.float64))
    xn = (x - weights["x_mean"]) / weights["x_std"]

    h1 = _act(xn @ weights["W1"].T + weights["b1"])
    h2 = _act(h1 @ weights["W2"].T + weights["b2"])
    yn = h2 @ weights["W3"].T + weights["b3"]

    y = yn * weights["y_std"] + weights["y_mean"]
    return y if y.shape[0] > 1 else y[0]

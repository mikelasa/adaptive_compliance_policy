"""
Export trained MLP weights for C++ Eigen inference.

Two artifacts are written:
  - <name>.npz : all arrays, for Python-side inspection / reload / parity checks.
  - <name>.bin : flat little-endian binary consumed by the C++ corrector.

Binary layout (all multi-byte values little-endian):
    int32   magic       = 0x57424D31  ("WBM1")
    int32   input_dim   = 3
    int32   hidden1     = 32
    int32   hidden2     = 16
    int32   output_dim  = 6
    float64 x_mean[3]
    float64 x_std [3]
    float64 y_mean[6]
    float64 y_std [6]
    float64 W1[32*3]    row-major, shape (hidden1, input_dim)
    float64 b1[32]
    float64 W2[16*32]   row-major, shape (hidden2, hidden1)
    float64 b2[16]
    float64 W3[6*16]    row-major, shape (output_dim, hidden2)
    float64 b3[6]

PyTorch Linear stores weight as (out, in) and computes y = W @ x + b, so the
matrices are written exactly as-is and the C++ side multiplies W * x directly.
"""

import struct
import numpy as np

MAGIC_V1 = 0x57424D31  # "WBM1" — legacy, relu assumed
MAGIC_V2 = 0x57424D32  # "WBM2" — includes activation field

# Activation encoding (stored as int32 in the binary)
ACT_IDS = {"relu": 0, "gelu": 1, "tanh": 2, "elu": 3}
ACT_NAMES = {v: k for k, v in ACT_IDS.items()}


def collect_weights(model, x_mean, x_std, y_mean, y_std, activation: str = "relu") -> dict:
    """Pull the three Linear layers out of the model into plain float64 arrays."""
    L1, L2, L3 = model.linear_layers()
    return dict(
        activation=np.array(ACT_IDS.get(activation, 0), dtype=np.int32),
        W1=L1.weight.detach().cpu().numpy().astype(np.float64),
        b1=L1.bias.detach().cpu().numpy().astype(np.float64),
        W2=L2.weight.detach().cpu().numpy().astype(np.float64),
        b2=L2.bias.detach().cpu().numpy().astype(np.float64),
        W3=L3.weight.detach().cpu().numpy().astype(np.float64),
        b3=L3.bias.detach().cpu().numpy().astype(np.float64),
        x_mean=np.asarray(x_mean, np.float64),
        x_std=np.asarray(x_std, np.float64),
        y_mean=np.asarray(y_mean, np.float64),
        y_std=np.asarray(y_std, np.float64),
    )


def save_npz(weights: dict, path: str):
    np.savez(path, **weights)


def save_bin(weights: dict, path: str):
    W1, W2, W3 = weights["W1"], weights["W2"], weights["W3"]
    hidden1, input_dim = W1.shape
    hidden2, _ = W2.shape
    output_dim, _ = W3.shape
    act_id = int(weights.get("activation", np.array(0)))

    with open(path, "wb") as f:
        f.write(struct.pack("<i", MAGIC_V2))
        f.write(struct.pack("<iiiii", input_dim, hidden1, hidden2, output_dim, act_id))

        def wr(a):
            f.write(np.ascontiguousarray(a, dtype="<f8").tobytes())

        wr(weights["x_mean"]); wr(weights["x_std"])
        wr(weights["y_mean"]); wr(weights["y_std"])
        wr(W1); wr(weights["b1"])
        wr(W2); wr(weights["b2"])
        wr(W3); wr(weights["b3"])


def export(model, x_mean, x_std, y_mean, y_std, out_prefix: str,
           activation: str = "relu") -> dict:
    """Collect weights, write both artifacts, return the weights dict."""
    weights = collect_weights(model, x_mean, x_std, y_mean, y_std, activation)
    save_npz(weights, out_prefix + ".npz")
    save_bin(weights, out_prefix + ".bin")
    print(f"[export] wrote {out_prefix}.npz and {out_prefix}.bin")
    return weights

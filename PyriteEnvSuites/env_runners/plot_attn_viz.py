"""
Plot modality attention weights captured during inference.
Handles both modality-attention and bi-cross-attention-DAT pkl formats.

Usage:
    python plot_attn_viz.py                     # loads from ./attn_viz/
    python plot_attn_viz.py <path/to/attn_viz>  # loads from given directory
"""

import sys
import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ── load ──────────────────────────────────────────────────────────────────────
attn_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "attn_viz")

files = sorted(
    glob.glob(os.path.join(attn_dir, "attn_weights_*.pkl")),
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[-1]),
)
assert files, f"No pkl files found in {attn_dir}"

data = []
for f in files:
    with open(f, "rb") as fh:
        data.append(pickle.load(fh))

mode = data[0].get("mode", "modality-attention")
steps = np.arange(len(data))
img_obs   = np.array([d["img_obs"]   for d in data])
force_obs = np.array([d["force_obs"] for d in data])

# ── smooth (sliding window, same as ImplicitRDP Fig. 6) ───────────────────────
def smooth(x, w=10):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")

steps_s      = steps[:len(smooth(img_obs))]
img_smooth   = smooth(img_obs)
force_smooth = smooth(force_obs)

# ── plot ──────────────────────────────────────────────────────────────────────
if mode == "modality-attention":
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    n_img = data[0]["n_img"]
    per_key = np.array([d["per_key"] for d in data])  # (N, 3)
    token_labels = [f"CLS frame {i}" for i in range(n_img)] + ["Force token"]
    colors_pk = ["steelblue", "cornflowerblue", "darkorange"]

    ax = axes[0]
    ax.plot(steps_s, img_smooth,   label="Image tokens", color="steelblue",  linewidth=2)
    ax.plot(steps_s, force_smooth, label="Force token",  color="darkorange", linewidth=2)
    ax.set_title("modality-attention — image vs force (smoothed, w=10)")
    ax.set_ylabel("Attention weight (avg)")
    ax.set_xlabel("Inference step")
    ax.set_ylim(0, 1); ax.legend(); ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    for i in range(per_key.shape[1]):
        ax2.plot(steps, per_key[:, i], label=token_labels[i],
                 color=colors_pk[i % len(colors_pk)], linewidth=1.5, alpha=0.8)
    ax2.set_title("Per-token attention (raw)")
    ax2.set_ylabel("Attention weight"); ax2.set_xlabel("Inference step")
    ax2.set_ylim(0, 1); ax2.legend(); ax2.grid(True, alpha=0.3)

elif mode in ("bi-cross-attention-DAT", "bi-cross-attention", "DAT"):
    has_cross       = "cross_per_force_step" in data[0]
    has_force_cross = "cross_per_img_token"  in data[0]
    n_rows = 2 + int(has_cross) + int(has_force_cross)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 4 * n_rows))
    axes = np.atleast_1d(axes)

    n_img   = data[0]["n_img"]
    n_force = data[0]["n_force"]
    ax_idx  = 0

    # --- [0] image vs force mass from attn-pool ---
    ax = axes[ax_idx]; ax_idx += 1
    ax.plot(steps_s, img_smooth,   label=f"Image tokens ({n_img})",   color="steelblue",  linewidth=2)
    ax.plot(steps_s, force_smooth, label=f"Force tokens ({n_force})", color="darkorange", linewidth=2)
    ax.set_title(f"{mode} — attn-pool: image vs force mass (smoothed, w=10)")
    ax.set_ylabel("Attention weight (sum over token group)")
    ax.set_xlabel("Inference step")
    ax.set_ylim(0, 1); ax.legend(); ax.grid(True, alpha=0.3)

    # --- [1] attn-pool average weight per token ---
    ax2 = axes[ax_idx]; ax_idx += 1
    pool_all = np.array([d["pool_per_token"] for d in data])  # (N, n_img+n_force+1)
    img_per_token   = pool_all[:, 1:1 + n_img].mean(axis=1)
    force_per_token = pool_all[:, 1 + n_img:].mean(axis=1)
    ax2.plot(steps, img_per_token,   label="Avg per image token",  color="steelblue",  alpha=0.8)
    ax2.plot(steps, force_per_token, label="Avg per force token",  color="darkorange", alpha=0.8)
    ax2.set_title("Attn-pool — average weight per token (raw)")
    ax2.set_ylabel("Avg weight per token"); ax2.set_xlabel("Inference step")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    # --- [2] img←force cross-attention heatmap (bi-cross modes only) ---
    if has_cross:
        cross_all = np.array([d["cross_per_force_step"] for d in data])  # (N, n_force)
        ax3 = axes[ax_idx]; ax_idx += 1
        im = ax3.imshow(cross_all.T, aspect="auto", origin="lower",
                        extent=[0, len(steps), 0, n_force],
                        cmap="hot", vmin=0)
        ax3.set_title("img←force cross-attention: which force timestep each image patch uses")
        ax3.set_xlabel("Inference step"); ax3.set_ylabel("Force timestep (0=oldest, 31=newest)")
        fig.colorbar(im, ax=ax3, label="Avg attention weight")

    # --- [3] force←img cross-attention heatmap (bi-cross modes only) ---
    if has_force_cross:
        force_cross_all = np.array([d["cross_per_img_token"] for d in data])  # (N, n_img)
        ax4 = axes[ax_idx]; ax_idx += 1
        im2 = ax4.imshow(force_cross_all.T, aspect="auto", origin="lower",
                         extent=[0, len(steps), 0, n_img],
                         cmap="hot", vmin=0)
        ax4.set_title("force←img cross-attention: which image patch each force timestep uses")
        ax4.set_xlabel("Inference step"); ax4.set_ylabel("Image token index (0=CLS, 1-49=frame0, 50-99=frame1)")
        fig.colorbar(im2, ax=ax4, label="Avg attention weight")

else:
    raise ValueError(f"Unknown mode in pkl files: {mode}")

plt.tight_layout()
out_path = os.path.join(attn_dir, "attn_viz.png")
plt.savefig(out_path, dpi=150)
print(f"Saved → {out_path}")
plt.show()

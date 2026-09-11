"""
Plot the bi-cross-attention module's OWN internal attention (img<-force and
force<-img), as opposed to plot_force_attn_correlation.py which plots the
DENOISER's cross-attention over fused obs tokens. This is a diagnostic on
what the fusion module itself is doing, one level upstream.

Consumes the same "transformer_dp" pkl schema, but reads the "encoder" field
instead of "denoiser_cross_per_cond_token". Only populated when
fuse_mode in ("bi-cross-attention", "bi-cross-attention-DAT") — see
TimmObsEncoderBiCrossDATTransformer._capture_cross_fusion_attn(). Plain "DAT"
mode has no cross-attention step, so there is nothing for this script to plot
for that fuse_mode.

Each "encoder" entry (kind == "cross") holds two proper probability
distributions (softmax weights, averaged over queries and heads, each
summing to ~1):
    cross_per_force_step: (n_force,) — which force timesteps the image
        patches attend to (query = image, key = force).
    cross_per_img_token:  (n_img,)   — which image patches the force tokens
        attend to (query = force, key = image).

Rather than plotting the raw distributions, this tracks their entropy per
step, converted to "effective number of tokens attended" (exp(entropy)):
1.0 means fully concentrated on a single token, N means spread uniformly
over all N. Falling effective-N during contact would mean the module is
narrowing in on specific force timesteps / image patches when it matters;
flat/high effective-N would mean it's attending broadly regardless of
contact state.

Usage:
    python plot_fusion_attn_correlation.py                     # loads ./attn_viz/
    python plot_fusion_attn_correlation.py <path/to/attn_viz>  # loads given dir
"""

import sys
import os
import glob
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

for _font_path in (
    "/Users/wendi/Library/Fonts/Calibri.ttf",
    "/Users/wendi/Library/Fonts/Calibri-Bold.ttf",
):
    if os.path.exists(_font_path):
        font_manager.fontManager.addfont(_font_path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=_font_path).get_name()

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def smooth_curve(data, window_size):
    """Same centered moving average as plot_force_attn_correlation.py."""
    smoothed = []
    for i in range(len(data)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(data), i + window_size // 2 + 1)
        smoothed.append(sum(data[start_idx:end_idx]) / (end_idx - start_idx))
    return smoothed


def effective_n(dist, eps=1e-12):
    """exp(entropy) of a probability vector — 'effective number of tokens
    attended'. 1.0 = fully concentrated, len(dist) = fully uniform."""
    dist = np.asarray(dist, dtype=np.float64)
    dist = dist / dist.sum()  # guard against tiny numerical drift off 1.0
    entropy = -np.sum(dist * np.log(dist + eps))
    return float(np.exp(entropy))


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
        d = pickle.load(fh)
        enc = d.get("encoder")
        if d.get("arch") == "transformer_dp" and enc is not None and enc.get("kind") == "cross":
            data.append(d)
assert data, (
    f"No transformer_dp entries with encoder-side cross-attention found in {attn_dir}. "
    "This only exists for fuse_mode in (bi-cross-attention, bi-cross-attention-DAT) — "
    "plain 'DAT' and 'modality-attention' have no bi-cross-attention step to capture."
)

mode = data[0]["mode"]

img_to_force_eff_n = []   # image patches attending over force timesteps
force_to_img_eff_n = []   # force tokens attending over image patches
wrench_norm = []
for d in data:
    enc = d["encoder"]
    img_to_force_eff_n.append(effective_n(enc["cross_per_force_step"]))
    force_to_img_eff_n.append(effective_n(enc["cross_per_img_token"]))
    wrench_norm.append(d["wrench_norm"] if d["wrench_norm"] is not None else np.nan)

window_size = 10
img_to_force_eff_n = smooth_curve(img_to_force_eff_n, window_size)
force_to_img_eff_n = smooth_curve(force_to_img_eff_n, window_size)
wrench_norm_smooth = smooth_curve(wrench_norm, window_size)

fig, (ax_attn, ax_force) = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)

l1, = ax_attn.plot(img_to_force_eff_n, label="Image -> Force (eff. N)", linewidth=3)
l2, = ax_attn.plot(force_to_img_eff_n, label="Force -> Image (eff. N)", linewidth=3)
ax_attn.set_xlabel("Steps", fontsize=22)
ax_attn.set_ylabel("Effective N attended", fontsize=22)
ax_attn.set_title(f"RACP ({mode}) — encoder bi-cross-attention concentration", fontsize=16)
ax_attn.tick_params(labelsize=18)
fig.legend([l1, l2], ["Image -> Force (eff. N)", "Force -> Image (eff. N)"],
           fontsize=16, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)

ax_force.plot(wrench_norm_smooth, color="firebrick", linewidth=3, label="||Wrench||")
ax_force.set_xlabel("Steps", fontsize=22)
ax_force.set_ylabel("||Wrench||", fontsize=22)
ax_force.tick_params(labelsize=18)

out_dir = os.path.join(attn_dir, "..", "figs", "attention")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "fusion_attention.pdf")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved -> {out_path}")

png_path = os.path.join(attn_dir, "fusion_attn_correlation.png")
plt.savefig(png_path, dpi=150, bbox_inches="tight")
print(f"Saved -> {png_path}")

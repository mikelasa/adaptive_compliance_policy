"""
Plot denoiser cross-attention mass (image tokens vs. force tokens) over a
rollout's control steps, alongside the measured wrench magnitude at each step.

Styled to match ImplicitRDP's own Fig. 7 plotting script (draw_attn_curve.py)
as closely as possible: same centered-window smoothing function, same
pdf/ps.fonttype embedding, same line widths / label sizes. Two additions
(per request) beyond ImplicitRDP's original: a second subplot with wrench
magnitude (display-smoothed, window=10, same as the attention curves), to
eyeball whether attention mass tracks contact; and a third subplot with the
same wrench_norm values completely unsmoothed, in case the display smoothing
hides something. Both come from the same source value per step
(wrench_norm in the pkl) -- the wrench itself is already unfiltered at
capture time (get_wrench() -> _wrench_buffers, the raw/unfiltered sensor
buffer, not the EMA-filtered _wrench_fb used only for haptic/eoat control
feedback); the only difference between the two panels is this script's own
display smoothing.

Consumes the "transformer_dp" pkl schema written by
DiffusionTransformerTimmMod1Policy._dump_attention_viz() — one pkl per
predict_action() call (i.e. per control step), each holding:
    denoiser_cross_per_cond_token: (n_cond_tokens,) attention mass captured at
        the FIRST decoder layer on the LAST denoising step only (see
        TransformerForActionDiffusion.pop_attention_viz_capture()) — this is
        the RACP-side counterpart of ImplicitRDP's `layer == 0 and
        timestep.item() == 0` snapshot, so the two are directly comparable.
    n_img_tokens / n_force_tokens / n_lowdim_tokens: token-count split used to
        slice denoiser_cross_per_cond_token into per-modality mass (the last
        entry, the diffusion-timestep token, is dropped — ImplicitRDP's own
        Fig. 7 doesn't plot its analogous "time" token either).
    wrench_norm: raw (unnormalized) wrench L2 norm at that control step.

Usage:
    python plot_force_attn_correlation.py                     # loads ./attn_viz/
    python plot_force_attn_correlation.py <path/to/attn_viz>  # loads given dir
"""

import sys
import os
import glob
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ── match ImplicitRDP's font/embedding setup, degrading gracefully if their
# Calibri files aren't present on this machine (they're macOS paths) ─────────
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
    """Verbatim port of ImplicitRDP's draw_attn_curve.py smoothing: a centered
    moving average with a shrinking window at the sequence edges."""
    smoothed = []
    for i in range(len(data)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(data), i + window_size // 2 + 1)
        smoothed.append(sum(data[start_idx:end_idx]) / (end_idx - start_idx))
    return smoothed


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
        if d.get("arch") == "transformer_dp" and d.get("denoiser_cross_per_cond_token") is not None:
            data.append(d)
assert data, (
    f"No transformer_dp entries with denoiser_cross_per_cond_token found in {attn_dir}. "
    "Was VISUALIZE_ATTENTION on during the rollout, and is this the mod1/transformer policy?"
)

mode = data[0]["mode"]
n_img = data[0]["n_img_tokens"]
n_force = data[0]["n_force_tokens"]
n_lowdim = data[0]["n_lowdim_tokens"]
assert n_img is not None and n_force is not None, (
    "Missing n_img_tokens/n_force_tokens in pkl — re-run inference after pulling "
    "the encoder token-layout-metadata fix."
)

# denoiser_cross_per_cond_token layout: [image][force][low_dim][timestep] — sum
# each modality's slice into a single per-step scalar, exactly like ImplicitRDP
# sums mean_attn[1:1+98] / mean_attn[1+102:] in transformer_for_diffusion.py.
slow_img_obs_attn = []
temporal_cond_attn = []  # naming matches ImplicitRDP's Fig. 7 legend key
wrench_norm = []
for d in data:
    tok = np.asarray(d["denoiser_cross_per_cond_token"])
    slow_img_obs_attn.append(float(tok[:n_img].sum()))
    temporal_cond_attn.append(float(tok[n_img:n_img + n_force].sum()))
    wrench_norm.append(d["wrench_norm"] if d["wrench_norm"] is not None else np.nan)

# ── smoothing (their exact function, w=10) ─────────────────────────────────────
window_size = 10
slow_img_obs_attn = smooth_curve(slow_img_obs_attn, window_size)
temporal_cond_attn = smooth_curve(temporal_cond_attn, window_size)
wrench_norm_smooth = smooth_curve(wrench_norm, window_size)

# ── plot: attention curve on top (their style), wrench magnitude below ────────
# Third panel is the RAW (unsmoothed) wrench_norm — same underlying values as
# the middle panel, just without the window=10 centered moving average. Both
# come from the same source (get_wrench() -> _wrench_buffers, unfiltered at
# the sensor level); the only difference between panels 2 and 3 is this
# script's own display smoothing.
fig, (ax_attn, ax_force, ax_force_raw) = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)

l1, = ax_attn.plot(slow_img_obs_attn, label="Slow Image Tokens", linewidth=3)
l2, = ax_attn.plot(temporal_cond_attn, label="Fast Force Tokens", linewidth=3)
ax_attn.set_xlabel("Steps", fontsize=22)
ax_attn.set_ylabel("Attention", fontsize=22)
ax_attn.set_title(f"RACP ({mode}) — denoiser cross-attention, layer 0, last denoising step", fontsize=16)
ax_attn.tick_params(labelsize=18)
fig.legend([l1, l2], ["Slow Image Tokens", "Fast Force Tokens"],
           fontsize=18, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)

ax_force.plot(wrench_norm_smooth, color="firebrick", linewidth=3, label="||Wrench|| (smoothed, w=10)")
ax_force.set_xlabel("Steps", fontsize=22)
ax_force.set_ylabel("||Wrench||\n(smoothed)", fontsize=18)
ax_force.set_title("Display-smoothed (centered moving average, w=10) — matches panel 1's x-axis alignment", fontsize=13)
ax_force.tick_params(labelsize=18)

ax_force_raw.plot(wrench_norm, color="firebrick", linewidth=1.5, alpha=0.8, label="||Wrench|| (raw)")
ax_force_raw.set_xlabel("Steps", fontsize=22)
ax_force_raw.set_ylabel("||Wrench||\n(raw)", fontsize=18)
ax_force_raw.set_title("Raw per-step values, no smoothing", fontsize=13)
ax_force_raw.tick_params(labelsize=18)

out_dir = os.path.join(attn_dir, "..", "figs", "attention")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "attention.pdf")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved -> {out_path}")

png_path = os.path.join(attn_dir, "attn_force_correlation.png")
plt.savefig(png_path, dpi=150, bbox_inches="tight")
print(f"Saved -> {png_path}")

# ---------------------------------------------------------------------------
# Force-attending visual curriculum, ported from FACTR
# (Liu, Li, Shaw, Tao, Salakhutdinov, Pathak — CMU, arXiv:2502.17432),
# https://github.com/jasonjzliu/factr, factr/utils.py — Apache-2.0.
#
# Two operators: latent-space 1D blur of vision tokens (their `space: latent`,
# used for RACP's first curriculum run) and pixel-space 2D blur of the raw
# image before the ViT (their `space: pixel`, the default in their own
# train_bc.yaml and what their reported results were trained with). The
# downsample operator (avg-pool + upsample, their `operator: downsample`
# alternative to blur) is left out of this pass; port it from the original
# file if a later ablation wants it.
# ---------------------------------------------------------------------------

import math
import numpy as np
import torch
import torch.nn.functional as F


def gaussian_2d_kernel(kernel_size: int, sigma: float, device=None, dtype=None) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=device, dtype=dtype)
    coords -= (kernel_size - 1) / 2.0
    x, y = torch.meshgrid(coords, coords, indexing="xy")
    kernel_2d = torch.exp(-0.5 * (x**2 + y**2) / sigma**2)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d


def gaussian_2d_smoothing(img: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Blur a batch of raw images before they reach the vision encoder.

    Args:
        img: (..., C, H, W)
        scale: Gaussian sigma. scale <= 0 is a no-op (identity).
    """
    if scale <= 0:
        return img

    sigma = scale
    kernel_size = max(3, 2 * math.ceil(3 * sigma) + 1)
    kernel_2d = gaussian_2d_kernel(kernel_size, sigma, device=img.device, dtype=img.dtype)
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)

    C = img.shape[-3]
    kernel_2d = kernel_2d.repeat(C, 1, 1, 1)  # (C, 1, kH, kW), depthwise
    padding = kernel_size // 2

    original_shape = img.shape
    spatial_shape = original_shape[-2:]
    batch_size = int(np.prod(original_shape[:-3]))
    img_reshaped = img.reshape(batch_size, C, *spatial_shape)

    blurred = F.conv2d(img_reshaped, kernel_2d, groups=C, padding=padding)
    return blurred.view(*original_shape)


def gaussian_1d_smoothing(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Blur each token's feature vector along the last dimension. Applied to
    vision tokens (B, N, D) *before* fusion, so it composes with every
    fuse_mode unchanged and never touches force tokens.

    Args:
        x: (..., feature_dim)
        scale: Gaussian sigma. scale <= 0 is a no-op (identity).
    """
    if scale <= 0:
        return x

    sigma = scale
    kernel_size = max(3, 2 * int(3 * sigma) + 1)
    half_size = (kernel_size - 1) // 2
    arange = torch.arange(-half_size, half_size + 1, device=x.device, dtype=x.dtype)
    kernel_1d = torch.exp(-0.5 * (arange / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_1d = kernel_1d.view(1, 1, -1)

    original_shape = x.shape
    feature_dim = x.shape[-1]
    batch_size = int(np.prod(x.shape[:-1]))
    x_reshaped = x.reshape(batch_size, 1, feature_dim)

    smoothed = F.conv1d(x_reshaped, kernel_1d, padding=half_size)
    return smoothed.view(*original_shape)


def get_scale(scheduler, start, end, cur_step, max_step, ratio=2 / 3):
    """
    Curriculum schedule for the blur scale. Holds `start` for the first
    `ratio` fraction of training, then decays to `end` over the remainder.
    """
    assert start >= end, "start scale must be >= end scale"
    assert 0 <= cur_step, "cur_step must be non-negative"
    assert max_step > 0, "max_step must be positive"
    cur_step = min(cur_step, max_step)

    if scheduler == "no":
        return 0.0

    t = cur_step / max_step
    if t <= ratio or scheduler == "const":
        return start

    t_rescaled = (t - ratio) / (1 - ratio)
    if scheduler == "linear":
        scale = start + t_rescaled * (end - start)
    elif scheduler == "cos":
        scale = end + 0.5 * (start - end) * (1 + np.cos(t_rescaled * np.pi))
    elif scheduler == "exp":
        scale = start * np.exp(-5 * t_rescaled)
    elif scheduler == "step":
        steps = 10
        step_index = int(t_rescaled * steps)
        scale = start + step_index * (end - start) / steps
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler}")

    return float(scale)

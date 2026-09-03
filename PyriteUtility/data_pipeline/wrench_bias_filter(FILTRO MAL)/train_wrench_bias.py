"""
Train the wrench-bias correction MLP.

Config-driven: all paths and hyperparameters live in a YAML file (see
config/train_wrench_bias_V1.yaml). Train/val loss are logged to Weights & Biases.

Pipeline:
    1. Build (xyz -> wrench) dataset from calibration episodes.
    2. Standardise inputs and outputs (stats from TRAIN split only).
    3. Train a tiny MLP (3 -> h1 -> h2 -> 6) with MSE + Adam.
    4. Report per-axis test RMSE and residual force-norm after correction.
    5. Export weights to .npz (inspection) and .bin (C++), with a NumPy parity check.

Usage:
    python train_wrench_bias.py --config config/train_wrench_bias_V1.yaml
    python train_wrench_bias.py --config config/train_wrench_bias_V1.yaml --epochs 800
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_PATH, "../../../"))

from PyriteUtility.data_pipeline.wrench_bias_filter.dataset import build_dataset
from PyriteUtility.data_pipeline.wrench_bias_filter.model import (
    WrenchBiasMLP,
    numpy_forward,
)
from PyriteUtility.data_pipeline.wrench_bias_filter import export as export_mod

AXES = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]


# ─── Config loading ───────────────────────────────────────────────────────────
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_cli_overrides(cfg: dict, args) -> dict:
    """Let a few common flags override the YAML for quick experiments."""
    if args.data is not None:
        cfg["paths"]["data_folder"] = args.data
    if args.out is not None:
        cfg["paths"]["output_dir"] = args.out
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.device is not None:
        cfg["training"]["device"] = args.device
    if args.no_wandb:
        cfg["logging"]["wandb"] = False
    return cfg


def standardise(arr):
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std[std < 1e-8] = 1.0  # guard against constant columns
    return mean, std


def train(cfg: dict):
    p_paths = cfg["paths"]
    p_ds = cfg["dataset"]
    p_model = cfg["model"]
    p_tr = cfg["training"]
    p_log = cfg["logging"]

    torch.manual_seed(p_tr["seed"])
    np.random.seed(p_tr["seed"])

    want_cuda = str(p_tr["device"]).startswith("cuda")
    device = p_tr["device"] if (want_cuda and torch.cuda.is_available()) else "cpu"

    os.makedirs(p_paths["output_dir"], exist_ok=True)
    out_prefix = os.path.join(p_paths["output_dir"], cfg["name"])

    # ── wandb ─────────────────────────────────────────────────────────────────
    use_wandb = bool(p_log.get("wandb", False))
    wandb_run = None
    if use_wandb:
        import wandb

        wandb_run = wandb.init(
            project=p_log["project"],
            entity=p_log.get("entity"),
            mode=p_log.get("mode", "online"),
            name=cfg["name"],
            tags=p_log.get("tags"),
            config=cfg,
        )

    # ── 1. Dataset ────────────────────────────────────────────────────────────
    X, Y, info = build_dataset(
        p_paths["data_folder"],
        robot_id=p_ds["robot_id"],
        window_frac=p_ds["window_frac"],
        max_wrench_std=p_ds["max_wrench_std"],
        max_pos_std_m=p_ds["max_pos_std_m"],
    )
    if len(X) < 20:
        raise RuntimeError(f"Only {len(X)} usable episodes — need more data.")

    if use_wandb:
        fn = np.linalg.norm(Y[:, :3], axis=1)
        wandb_run.summary.update(
            {
                "n_episodes_total": info["n_total"],
                "n_episodes_accepted": info["n_accepted"],
                "n_episodes_rejected": info["n_rejected"],
                "bias_force_norm_mean_raw": float(fn.mean()),
                "bias_force_norm_max_raw": float(fn.max()),
            }
        )

    # ── 2. Split + standardise (stats on TRAIN ONLY) ──────────────────────────
    n = len(X)
    perm = np.random.permutation(n)
    n_test = max(1, int(p_ds["test_frac"] * n))
    n_val = max(1, int(p_ds["val_frac"] * n))
    test_idx = perm[:n_test]
    val_idx = perm[n_test : n_test + n_val]
    train_idx = perm[n_test + n_val :]

    x_mean, x_std = standardise(X[train_idx])
    y_mean, y_std = standardise(Y[train_idx])

    def to_tensor(idx):
        xn = (X[idx] - x_mean) / x_std
        yn = (Y[idx] - y_mean) / y_std
        return (
            torch.tensor(xn, dtype=torch.float32, device=device),
            torch.tensor(yn, dtype=torch.float32, device=device),
        )

    Xtr, Ytr = to_tensor(train_idx)
    Xva, Yva = to_tensor(val_idx)
    print(f"[train] split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")

    # ── 3. Train ──────────────────────────────────────────────────────────────
    input_dim = X.shape[1]
    print(f"[train] input_dim={input_dim}  ({input_dim} features per sample)")
    model = WrenchBiasMLP(
        input_dim, p_model["hidden1"], p_model["hidden2"],
        p_model.get("dropout", 0.0), p_model.get("activation", "relu"),
    ).to(device)
    opt = torch.optim.Adam(
        model.parameters(), lr=p_tr["lr"], weight_decay=p_tr["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=p_tr["patience"] // 3, min_lr=1e-6
    )
    if p_tr.get("loss", "mse") == "huber":
        loss_fn = nn.HuberLoss(delta=p_tr.get("huber_delta", 1.0))
    else:
        loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    patience_ctr = 0
    bs = p_tr["batch_size"]
    train_losses, val_losses = [], []

    for epoch in range(p_tr["epochs"]):
        model.train()
        idx = torch.randperm(len(Xtr), device=device)
        epoch_losses = []
        for i in range(0, len(Xtr), bs):
            b = idx[i : i + bs]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[b]), Ytr[b])
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())
        train_loss = float(np.mean(epoch_losses))

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva), Yva).item()

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        current_lr = opt.param_groups[0]["lr"]
        if use_wandb:
            wandb_run.log(
                {"train/loss": train_loss, "val/loss": val_loss, "lr": current_lr, "epoch": epoch}
            )

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if epoch % 20 == 0:
            print(f"[train] epoch {epoch:4d}  train={train_loss:.5f}  val={val_loss:.5f}  lr={current_lr:.2e}")
        if patience_ctr >= p_tr["patience"]:
            print(f"[train] early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ── 4. Test-set evaluation in PHYSICAL units ──────────────────────────────
    model.eval()
    with torch.no_grad():
        xn = torch.tensor((X[test_idx] - x_mean) / x_std, dtype=torch.float32, device=device)
        pred = model(xn).cpu().numpy() * y_std + y_mean
    target = Y[test_idx]

    per_axis_rmse = np.sqrt(((pred - target) ** 2).mean(axis=0))
    print("\n[eval] per-axis test RMSE (physical units):")
    for a, r in zip(AXES, per_axis_rmse):
        print(f"[eval]   {a}: {r:.4f}")

    resid = target - pred
    fn_before = np.linalg.norm(target[:, :3], axis=1)
    fn_after = np.linalg.norm(resid[:, :3], axis=1)
    print(f"\n[eval] force-norm  before: mean={fn_before.mean():.3f} N  max={fn_before.max():.3f} N")
    print(f"[eval] force-norm  after : mean={fn_after.mean():.3f} N  max={fn_after.max():.3f} N")

    if use_wandb:
        wandb_run.summary.update(
            {f"test/rmse_{a}": float(r) for a, r in zip(AXES, per_axis_rmse)}
        )
        wandb_run.summary.update(
            {
                "test/force_norm_before_mean": float(fn_before.mean()),
                "test/force_norm_after_mean": float(fn_after.mean()),
                "test/force_norm_before_max": float(fn_before.max()),
                "test/force_norm_after_max": float(fn_after.max()),
                "best_val_loss": best_val,
            }
        )

    # ── 5. Export + NumPy parity check ────────────────────────────────────────
    weights = export_mod.export(model.cpu(), x_mean, x_std, y_mean, y_std, out_prefix,
                               activation=p_model.get("activation", "relu"))

    with torch.no_grad():
        torch_out = (
            model(torch.tensor((X - x_mean) / x_std, dtype=torch.float32)).numpy()
            * y_std
            + y_mean
        )
    numpy_out = numpy_forward(weights, X)
    max_diff = np.abs(torch_out - numpy_out).max()
    print(f"\n[verify] max |torch - numpy| over all samples: {max_diff:.2e}")
    if max_diff > 1e-4:
        print("[verify] WARNING: parity check exceeded 1e-4 — check export layout.")
    else:
        print("[verify] OK — exported weights reproduce the trained model.")

    if p_log.get("plot", False):
        loss_png = _plot_losses(train_losses, val_losses, out_prefix)
        scatter_png = _plot(target, pred, out_prefix)
        if use_wandb:
            import wandb
            if loss_png:
                wandb_run.log({"train/loss_curve": wandb.Image(loss_png)})
            if scatter_png:
                wandb_run.log({"test/scatter": wandb.Image(scatter_png)})

    if use_wandb:
        wandb_run.finish()


def _plot_losses(train_losses, val_losses, out_prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(len(train_losses))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_losses, label="train")
    ax.plot(epochs, val_losses,   label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss (normalised)")
    ax.set_title("Train vs validation loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png = out_prefix + "_loss_curve.png"
    fig.savefig(png, dpi=120)
    plt.close(fig)
    print(f"[plot] saved {png}")
    return png


def _plot(target, pred, out_prefix):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    for k in range(6):
        a = ax[k // 3][k % 3]
        a.scatter(target[:, k], pred[:, k], s=12, alpha=0.7)
        lo = min(target[:, k].min(), pred[:, k].min())
        hi = max(target[:, k].max(), pred[:, k].max())
        a.plot([lo, hi], [lo, hi], "r--", lw=1)
        a.set_title(AXES[k])
        a.set_xlabel("measured bias")
        a.set_ylabel("predicted bias")
    fig.suptitle("Wrench bias: predicted vs measured (test set)")
    fig.tight_layout()
    png = out_prefix + "_test_scatter.png"
    fig.savefig(png, dpi=120)
    print(f"[plot] saved {png}")
    return png


def main():
    default_cfg = os.path.join(SCRIPT_PATH, "config", "train_wrench_bias.yaml")
    p = argparse.ArgumentParser(description="Train wrench-bias correction MLP")
    p.add_argument("--config", default=default_cfg, help="path to YAML config")
    # optional quick overrides
    p.add_argument("--data", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default=None, help='e.g. "cpu" or "cuda:0"')
    p.add_argument("--no_wandb", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)
    train(cfg)


if __name__ == "__main__":
    main()

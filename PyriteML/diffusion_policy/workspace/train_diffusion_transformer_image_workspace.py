import sys
import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.append(ROOT_DIR)

if __name__ == "__main__":
    os.chdir(ROOT_DIR)

import copy
import logging
import pickle
import random

import hydra
import numpy as np
import torch
import tqdm
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.dataset.base_dataset import BaseDataset, BaseImageDataset
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.policy.diffusion_transformer_timm_mod1_policy import DiffusionTransformerTimmMod1Policy
from diffusion_policy.workspace.base_workspace import BaseWorkspace

logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainDiffusionTransformerImageWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionTransformerTimmMod1Policy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionTransformerTimmMod1Policy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # ── optimizer param groups ────────────────────────────────────────────
        # Three groups so that all parameters are actually optimized:
        #   1. transformer denoiser — full LR
        #   2. pretrained vision backbone — reduced LR (it is already pretrained)
        #   3. everything else in obs_encoder (force enc, cross-attn, DAT, …) — full LR
        obs_encoder_lr = cfg.optimizer.lr
        if cfg.policy.obs_encoder["reduce_pretrained_lr"]:
            obs_encoder_lr *= 0.1
            logger.info("==> reduce pretrained obs_encoder lr to %.2e", obs_encoder_lr)

        pretrained_param_ids = set()
        pretrained_params = []
        logger.info("==> rgb keys: %s", self.model.obs_encoder.rgb_keys)
        for key in self.model.obs_encoder.rgb_keys:
            for param in self.model.obs_encoder.key_model_map[key].parameters():
                if param.requires_grad and id(param) not in pretrained_param_ids:
                    pretrained_params.append(param)
                    pretrained_param_ids.add(id(param))
        logger.info("pretrained backbone params: %d", len(pretrained_params))

        other_obs_params = [
            p for p in self.model.obs_encoder.parameters()
            if p.requires_grad and id(p) not in pretrained_param_ids
        ]
        logger.info("other obs_encoder params: %d", len(other_obs_params))

        optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
        optimizer_cfg.pop("_target_")

        # Use the denoiser's own get_optim_groups so that pos_emb, cond_pos_emb,
        # LayerNorm weights, and biases are correctly excluded from weight decay.
        denoiser_groups = self.model.model.get_optim_groups(
            weight_decay=optimizer_cfg["weight_decay"]
        )
        param_groups = [
            *denoiser_groups,
            {"params": pretrained_params, "lr": obs_encoder_lr},
            {"params": other_obs_params},   # full LR: force enc, cross-attn, DAT, …
        ]
        self.optimizer = torch.optim.AdamW(params=param_groups, **optimizer_cfg)

        # configure training state
        self.global_step = 0
        self.epoch = 0

        if not cfg.training.resume:
            self.exclude_keys = ["optimizer"]

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # find_unused_parameters=True is needed for transformer models where
        # not every parameter contributes to every forward pass
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            log_with="wandb",
            kwargs_handlers=[ddp_kwargs],
        )
        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb_cfg.pop("project")
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg},
        )

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                accelerator.print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset) or isinstance(dataset, BaseDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)

        # compute normalizer on the main process and save to disk
        sparse_normalizer_path = os.path.join(self.output_dir, "sparse_normalizer.pkl")
        if accelerator.is_main_process:
            sparse_normalizer = dataset.get_normalizer()
            pickle.dump(sparse_normalizer, open(sparse_normalizer_path, "wb"))

        accelerator.wait_for_everyone()
        sparse_normalizer = pickle.load(open(sparse_normalizer_path, "rb"))

        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        logger.info(
            "train dataset: %d, train dataloader: %d", len(dataset), len(train_dataloader)
        )
        logger.info(
            "val dataset: %d, val dataloader: %d", len(val_dataset), len(val_dataloader)
        )

        self.model.set_normalizer(sparse_normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(sparse_normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs
            ) // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1,
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )

        train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler = (
            accelerator.prepare(
                train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler
            )
        )

        # print sample batch info
        batch_size = cfg.dataloader.batch_size
        logger.info("batch_size: %d", batch_size)
        sample_batch = next(iter(train_dataloader))
        for key, attr in sample_batch["obs"]["sparse"].items():
            logger.info("obs.sparse.%s: %s", key, attr.shape)
        logger.info("action.sparse: %s", sample_batch["action"]["sparse"].shape)
        logger.info("dataset.action_type: %s", dataset.action_type)
        action_dimension = sample_batch["action"]["sparse"].shape[-1]

        device = self.model.device
        if self.ema_model is not None:
            self.ema_model.to(device)

        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        best_val_metric = float("inf")
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                self.model.train()

                step_log = dict()

                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = list()
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                        if (
                            batch_idx == 0
                            or batch["action"]["sparse"].shape[0] == batch_size
                        ):
                            train_sampling_batch = batch

                        raw_loss = self.model(batch)
                        accelerator.backward(raw_loss)

                        # optional gradient norm logging
                        if cfg.training.log_gradient_norm:
                            model_unwrapped = accelerator.unwrap_model(self.model)
                            grads = [
                                p.grad.detach().flatten()
                                for p in model_unwrapped.model.parameters()
                                if p.grad is not None
                            ]
                            grad_norm = torch.cat(grads).norm() if grads else 0.0
                            step_log["grad_norm"] = grad_norm

                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        if self.global_step % 500 == 0:
                            torch.cuda.empty_cache()

                        if cfg.training.use_ema:
                            ema.step(accelerator.unwrap_model(self.model))

                        raw_loss_cpu = raw_loss.detach().cpu().item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }

                        is_last_batch = batch_idx == (len(train_dataloader) - 1)
                        if not is_last_batch:
                            accelerator.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) and batch_idx >= (
                            cfg.training.max_train_steps - 1
                        ):
                            break

                train_loss = np.mean(train_losses)
                step_log["train_loss"] = train_loss

                # ========= eval for this epoch ==========
                policy = accelerator.unwrap_model(self.model)
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                def log_action_mse(step_log, category, pred_action, gt_action):
                    pred_naction = {
                        "sparse": sparse_normalizer["action"].normalize(
                            pred_action["sparse"]
                        ),
                    }
                    gt_naction = {
                        "sparse": sparse_normalizer["action"].normalize(
                            gt_action["sparse"]
                        ),
                    }
                    B, T, _ = pred_naction["sparse"].shape
                    pred_naction_sparse = pred_naction["sparse"].view(
                        B, T, -1, action_dimension
                    )
                    gt_naction_sparse = gt_naction["sparse"].view(
                        B, T, -1, action_dimension
                    )
                    step_log[f"{category}_sparse_naction_mse_error"] = (
                        torch.nn.functional.mse_loss(pred_naction_sparse, gt_naction_sparse)
                    )
                    step_log[f"{category}_sparse_cmd_naction_mse_error"] = (
                        torch.nn.functional.mse_loss(
                            pred_naction_sparse[..., :9], gt_naction_sparse[..., :9]
                        )
                    )
                    step_log[f"{category}_sparse_vt_naction_mse_error"] = (
                        torch.nn.functional.mse_loss(
                            pred_naction_sparse[..., 9:18], gt_naction_sparse[..., 9:18]
                        )
                    )
                    step_log[f"{category}_sparse_stiffness_mse_error"] = (
                        torch.nn.functional.mse_loss(
                            pred_naction_sparse[..., 18], gt_naction_sparse[..., 18]
                        )
                    )

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0 and accelerator.is_main_process:
                    with torch.no_grad():
                        batch = dict_apply(
                            train_sampling_batch,
                            lambda x: x.to(device, non_blocking=True),
                        )
                        gt_action = batch["action"]
                        pred_action = policy.predict_action(batch["obs"])
                        log_action_mse(step_log, "train", pred_action, gt_action)

                        if len(val_dataloader) > 0:
                            val_sampling_batch = next(iter(val_dataloader))
                            batch = dict_apply(
                                val_sampling_batch,
                                lambda x: x.to(device, non_blocking=True),
                            )
                            gt_action = batch["action"]
                            pred_action = policy.predict_action(batch["obs"])
                            log_action_mse(step_log, "val", pred_action, gt_action)

                            best_val_metric = min(
                                best_val_metric,
                                step_log["val_sparse_naction_mse_error"].item(),
                            )

                        del batch, gt_action, pred_action

                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0 and accelerator.is_main_process:
                    model_ddp = self.model
                    self.model = accelerator.unwrap_model(self.model)

                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    metric_dict = {k.replace("/", "_"): v for k, v in step_log.items()}
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                    self.model = model_ddp

                accelerator.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

        accelerator.end_training()
        return best_val_metric


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = TrainDiffusionTransformerImageWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()

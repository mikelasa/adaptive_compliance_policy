from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.transformer_for_action_diffusion import TransformerForActionDiffusion
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.vision.timm_obs_encoder_bi_cross_dat_transformer import TimmObsEncoderBiCrossDATTransformer


class DiffusionTransformerTimmMod1Policy(BaseImagePolicy):
    def __init__(self,
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: TimmObsEncoderBiCrossDATTransformer,
            num_inference_steps=None,
            input_pertub=0.1,
            # arch
            n_layer=7,
            n_head=8,
            n_emb=768,
            p_drop_attn=0.1,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shapes — same paths as DiffusionUnetTimmMod1Policy
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_horizon = shape_meta["sample"]["action"]["sparse"]["horizon"]

        obs_shape = obs_encoder.output_shape()
        assert len(obs_shape) == 3, (
            f"obs_encoder must return (1, N, D) tokens, got shape {obs_shape}. "
            "Use TimmObsEncoderBiCrossDATTransformer, not a flat-output encoder."
        )
        assert obs_shape[-1] == n_emb, (
            f"obs_encoder output dim {obs_shape[-1]} != n_emb {n_emb}. "
            "Set n_emb to match the vision encoder embed_dim (e.g. 768 for ViT-B)."
        )
        obs_tokens = obs_shape[-2]

        model = TransformerForActionDiffusion(
            input_dim=action_dim,
            output_dim=action_dim,
            action_horizon=action_horizon,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            max_cond_tokens=obs_tokens + 1,  # obs tokens + 1 diffusion-timestep token
            p_drop_attn=p_drop_attn,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.sparse_normalizer = LinearNormalizer()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.input_pertub = input_pertub
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========= training  ============
    def set_normalizer(self, sparse_normalizer: LinearNormalizer):
        self.sparse_normalizer.load_state_dict(sparse_normalizer.state_dict())

    def get_normalizer(self):
        return self.sparse_normalizer

    # ========= inference  ============
    def conditional_sample(self,
            condition_data, condition_mask,
            cond=None, generator=None,
            **kwargs):
        model = self.model
        scheduler = self.noise_scheduler

        if hasattr(model, "reset_attention_viz_capture"):
            model.reset_attention_viz_capture()

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = model(trajectory, t, cond)
            trajectory = scheduler.step(
                model_output, t, trajectory,
                generator=generator,
                **kwargs
            ).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action(self, obs: Dict) -> Dict[str, torch.Tensor]:
        """
        obs: dict with key "sparse" containing the flat obs dict.
             Matches the same interface as DiffusionUnetTimmMod1Policy.
        """
        obs_dict_sparse = obs["sparse"]
        nobs_sparse = self.sparse_normalizer.normalize(obs_dict_sparse)
        B = next(iter(nobs_sparse.values())).shape[0]

        # raw (unnormalized) wrench magnitude at the most recent timestep, for the
        # attention-viz force subplot only — has no effect on the policy itself.
        wrench_norm = None
        wrench_keys = getattr(self.obs_encoder, "wrench_keys", [])
        if wrench_keys:
            key = wrench_keys[0]
            # (B, T, 6) -> most recent timestep, batch 0 -> scalar L2 norm
            wrench_norm = float(
                torch.linalg.norm(obs_dict_sparse[key][0, -1].float()).item()
            )

        obs_tokens = self.obs_encoder(nobs_sparse)  # (B, N, n_emb)
        enc_capture = None
        if hasattr(self.obs_encoder, "pop_fusion_attention_viz"):
            enc_capture = self.obs_encoder.pop_fusion_attention_viz()

        cond_data = torch.zeros(
            size=(B, self.action_horizon, self.action_dim),
            device=self.device, dtype=self.dtype,
        )
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        nsample = self.conditional_sample(
            condition_data=cond_data,
            condition_mask=cond_mask,
            cond=obs_tokens,
            **self.kwargs,
        )

        dec_capture = None
        if hasattr(self.model, "pop_attention_viz_capture"):
            dec_capture = self.model.pop_attention_viz_capture()
        if enc_capture is not None or dec_capture is not None:
            self._dump_attention_viz(enc_capture, dec_capture, wrench_norm)

        assert nsample.shape == (B, self.action_horizon, self.action_dim)
        action_pred = self.sparse_normalizer["action"].unnormalize(nsample)

        return {"sparse": action_pred}

    def _dump_attention_viz(self, enc_capture, dec_capture, wrench_norm=None):
        """Pickle one combined attention snapshot per predict_action() call:
        the encoder-side bi-cross-attention (img<->force, or self-attention for
        modality-attention, or None for plain DAT) plus the denoiser-side
        cross-attention (action-horizon queries -> obs tokens + timestep token,
        captured at the first decoder layer on the last denoising step only —
        matches ImplicitRDP's Fig. 7 methodology, see
        TransformerForActionDiffusion.pop_attention_viz_capture()), plus the raw
        wrench magnitude at the current control step (for the force subplot in
        plot_force_attn_correlation.py; has no bearing on the policy itself).
        Also records the [image][force][low_dim] token-count split of
        denoiser_cross_per_cond_token so a plotting script can sum attention
        mass per modality without hardcoding indices (cf. ImplicitRDP's fixed
        slice indices in transformer_for_diffusion.py). Mirrors the
        pooled-encoder dump format in timm_obs_encoder_bi_cross_dat_V1.py but
        with arch="transformer_dp" and no attn-pool fields, since this pipeline
        has no pooling step.
        """
        import os
        import pickle

        attn_dict = {
            "arch": "transformer_dp",
            "mode": self.obs_encoder.fuse_mode,
            "encoder": enc_capture,                        # see pop_fusion_attention_viz()
            "denoiser_cross_per_cond_token": dec_capture,   # see pop_attention_viz_capture()
            "n_img_tokens": getattr(self.obs_encoder, "n_img_tokens", None),
            "n_force_tokens": getattr(self.obs_encoder, "n_force_tokens", None),
            "n_lowdim_tokens": getattr(self.obs_encoder, "n_lowdim_tokens", None),
            "wrench_norm": wrench_norm,
        }
        if not hasattr(self, "_attn_viz_count"):
            self._attn_viz_count = 0
        out_dir = getattr(self, "attn_viz_dir", "attn_viz")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"attn_weights_{self._attn_viz_count}.pkl"), "wb") as f:
            pickle.dump(attn_dict, f)
        self._attn_viz_count += 1

    def compute_loss(self, batch):
        assert "valid_mask" not in batch

        nobs_sparse = self.sparse_normalizer.normalize(batch["obs"]["sparse"])
        nactions = self.sparse_normalizer["action"].normalize(batch["action"]["sparse"])
        trajectory = nactions

        obs_tokens = self.obs_encoder(nobs_sparse)  # (B, N, n_emb)

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        # input perturbation to alleviate exposure bias — https://github.com/forever208/DDPM-IP
        noise_new = noise + self.input_pertub * torch.randn(
            trajectory.shape, device=trajectory.device
        )

        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (nactions.shape[0],), device=trajectory.device,
        ).long()

        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise_new, timesteps)

        pred = self.model(noisy_trajectory, timesteps, cond=obs_tokens)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = reduce(loss, "b ... -> b (...)", "mean")
        return loss.mean()

    def forward(self, batch):
        return self.compute_loss(batch)

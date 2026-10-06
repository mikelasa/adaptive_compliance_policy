import torch
import torch.nn as nn

from diffusion_policy.model.vision.timm_obs_encoder_with_force import TimmObsEncoderWithForce
from diffusion_policy.model.vision.curriculum import gaussian_2d_smoothing


class TimmObsEncoderWithForceCurriculumGated(TimmObsEncoderWithForce):
    """
    TimmObsEncoderWithForce + pixel-space force-attending visual curriculum,
    modality dropout, and the Schmitt-trigger contact gate -- ported from
    TimmObsEncoderBiCrossDATTransformer (see that file for the full design
    rationale in comments). Kept as a separate subclass rather than editing
    TimmObsEncoderWithForce in place, since that class is shared by several
    other active UNet baseline configs (train_conv_workspace_bi_cross*.yaml,
    train_conv_workspace_cross.yaml, train_conv_workspace.yaml).

    Only latent-space curriculum is omitted (never used on the RACP side
    either). The wrench branch here collapses to a single (B, D) feature
    per forward() call (no per-timestep token sequence like the DAT
    transformer encoder), so the gate blends that single vector with
    force_no_contact_token instead of broadcasting over a token dimension.
    """

    def __init__(
        self,
        shape_meta: dict,
        fuse_mode: str,
        reduce_pretrained_lr: bool,
        vision_encoder_cfg: dict,
        force_encoder_cfg: dict,
        position_encoding: str = "learnable",
        contact_gate_enabled: bool = False,
        contact_gate_threshold: float = 7.0,
        contact_gate_threshold_high: float = 17.0,
    ):
        super().__init__(
            shape_meta=shape_meta,
            fuse_mode=fuse_mode,
            reduce_pretrained_lr=reduce_pretrained_lr,
            vision_encoder_cfg=vision_encoder_cfg,
            force_encoder_cfg=force_encoder_cfg,
            position_encoding=position_encoding,
        )

        # set by the training workspace loop, train-only, no-op at eval
        self.curriculum_scale = 0.0
        self.curriculum_space = "pixel"
        self.img_dropout_p = 0.0
        self.force_dropout_p = 0.0

        # constructor arg, not runtime-settable: decides whether the model
        # even has the learnable no-contact placeholder parameter, so it
        # must be fixed at construction time for checkpoint state_dict
        # compatibility. Active at train AND eval (part of the policy).
        self.contact_gate_enabled = contact_gate_enabled
        self.contact_gate_threshold = contact_gate_threshold
        self.contact_gate_threshold_high = contact_gate_threshold_high
        if self.contact_gate_enabled:
            self.force_no_contact_token = nn.Parameter(torch.zeros(1, self.v_feature_dim))
            nn.init.normal_(self.force_no_contact_token, std=0.02)

        # set by the policy via set_wrench_normalizer(), so the gate can
        # unnormalize wrench back to raw Newtons before comparing against
        # contact_gate_threshold/_high. See set_wrench_normalizer() below.
        self._wrench_normalizer_ref = None
        self._wrench_normalizer_key = None

    def set_wrench_normalizer(self, sparse_normalizer, wrench_key):
        self._wrench_normalizer_ref = sparse_normalizer
        self._wrench_normalizer_key = wrench_key

    def forward(self, obs_dict):
        """Assume each image key is (B, T, C, H, W)"""
        features = list()
        modality_features = list()
        low_dim_features = list()
        batch_size = next(iter(obs_dict.values())).shape[0]

        # ── modality dropout (train-only) ───────────────────────────────────
        drop_img = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        drop_force = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        if self.training and (self.img_dropout_p > 0 or self.force_dropout_p > 0):
            drop_img = torch.rand(batch_size, device=self.device) < self.img_dropout_p
            drop_force = torch.rand(batch_size, device=self.device) < self.force_dropout_p
            drop_force = drop_force & ~drop_img  # never drop both for the same sample

        # process rgb input
        for key in self.rgb_keys:
            img = obs_dict[key]
            B, T = img.shape[:2]
            assert B == batch_size
            assert img.shape[2:] == self.key_shape_map[key]
            if drop_img.any():
                img = img.clone()
                img[drop_img] = 0.0
            img = img.reshape(B * T, *img.shape[2:])
            img = self.key_transform_map[key](img)

            # force-attending visual curriculum, pixel-space variant: degrade
            # the image before the (trainable) vision encoder ever sees it.
            if self.training and self.curriculum_scale > 0 and self.curriculum_space == "pixel":
                img = gaussian_2d_smoothing(img, self.curriculum_scale)

            raw_feature = self.key_model_map[key](img)
            feature = self.aggregate_feature(
                model_name=self.vision_encoder_cfg.model_name,
                agg_mode=self.vision_encoder_cfg.feature_aggregation,
                feature=raw_feature,
            )
            assert len(feature.shape) == 2 and feature.shape[0] == B * T
            features.append(feature.reshape(B, -1))
            modality_features.append(feature.reshape(B, T, -1))

        for key in self.wrench_keys:
            data = obs_dict[key]  # (B, T, 6)
            B, T = data.shape[:2]
            assert B == batch_size
            assert data.shape[2:] == self.key_shape_map[key]

            # ── contact gate (train + eval). Computed before modality
            # dropout below, so the two mechanisms never see each other.
            # Schmitt-trigger hysteresis scanned sequentially across the T
            # steps of the already-observed wrench window -- see
            # TimmObsEncoderBiCrossDATTransformer.forward() for the full
            # rationale (noise-floor/threshold straddling, t=0 seeding off
            # the high bound like every other step, etc). Unlike that
            # encoder, this one has no per-timestep force token sequence
            # to broadcast the gate over -- feature below is already a
            # single (B, D) vector per forward() call, so alpha blends
            # that single vector with the placeholder directly.
            if self.contact_gate_enabled:
                if self._wrench_normalizer_ref is not None:
                    raw = self._wrench_normalizer_ref[self._wrench_normalizer_key].unnormalize(data)
                else:
                    raw = data  # normalize_wrench: False for this task -- already raw
                force_norm = raw[:, :, :3].norm(dim=-1)  # (B, T)
                state = (force_norm[:, 0] > self.contact_gate_threshold_high).float()  # (B,)
                for t in range(1, force_norm.shape[1]):
                    ft = force_norm[:, t]
                    opens = ft > self.contact_gate_threshold_high
                    closes = ft < self.contact_gate_threshold
                    state = torch.where(opens, torch.ones_like(state), state)
                    state = torch.where(closes, torch.zeros_like(state), state)
                alpha = state  # (B,) gate value at the most recent timestep

            if drop_force.any():
                data = data.clone()
                data[drop_force] = 0.0

            data = data.permute(0, 2, 1)
            feature = self.key_model_map[key](data.float())[:, :, 0]
            assert len(feature.shape) == 2 and feature.shape[0] == B

            if self.contact_gate_enabled:
                a = alpha[:, None]
                placeholder = self.force_no_contact_token.expand(B, -1)
                feature = a * feature + (1.0 - a) * placeholder

            features.append(feature.reshape(B, -1))
            modality_features.append(feature.unsqueeze(1))

        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            B, T = data.shape[:2]
            assert B == batch_size
            assert data.shape[2:] == self.key_shape_map[key]
            features.append(data.reshape(B, -1))
            low_dim_features.append(data.reshape(B, -1))

        # concatenate all features
        if self.fuse_mode == "concat":
            result = torch.cat(features, dim=-1)
        elif self.fuse_mode == "mlp":
            result = self.mlp(torch.cat(modality_features, dim=-1))
            result = torch.concat([result, torch.cat(low_dim_features, dim=-1)], dim=1)
        elif self.fuse_mode == "modality-attention":
            in_embeds = torch.cat(modality_features, dim=1)  # [batch, n_features, D]
            if self.position_encoding == "learnable":
                if self.position_embedding.device != in_embeds.device:
                    self.position_embedding = self.position_embedding.to(in_embeds.device)
                in_embeds = in_embeds + self.position_embedding
            out_embeds = self.transformer_encoder(in_embeds)  # [batch, n_features, D]
            result = torch.concat(
                [out_embeds[:, i] for i in range(out_embeds.shape[1])], dim=1
            )
            result = self.linear_projection(result)
            result = torch.concat([result, torch.cat(low_dim_features, dim=-1)], dim=1)

        return result

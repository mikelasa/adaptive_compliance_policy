import copy

import timm
import torch
import torch.nn as nn
import torchvision
import logging

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import replace_submodules
from diffusion_policy.model.vision.utils.cross_attention import CrossAttention
from diffusion_policy.model.vision.utils.attention_pool import AttentionPool1d

from multimodal_representation.multimodal.models.base_models.encoders import (
    ForceEncoder,
)

from dual_attention.dual_attn_blocks import DualAttnEncoderBlock
from dual_attention.symbol_retrieval import (
    PositionalSymbolRetriever,
    PositionRelativeSymbolRetriever,
    SymbolicAttention,
    RelationalSymbolicAttention,
)

logger = logging.getLogger(__name__)


class TimmObsEncoderWithForceV2(ModuleAttrMixin):
    def __init__(
        self,
        shape_meta: dict,
        fuse_mode: str,
        reduce_pretrained_lr: bool,
        vision_encoder_cfg: dict,
        force_encoder_cfg: dict,
        position_encoding: str = "learnable",
        use_relational_features: bool = False,
        symbol_retriever: str = "positional",  # positional | position_relative | symbolic | relational_symbolic
        symbol_retriever_cfg: dict = None,
        bi_cross_heads: int = 4,
        n_heads_sa: int = 4,
        n_heads_ra: int = 4,
        share_attn_params: bool = False,  # whether to share parameters between SA and RA in the feature aggregation module
        dat_dff: int = 2048,
        dat_activation: str = "relu",
        dat_dropout_rate: float = 0.0,
        dat_norm_first: bool = True,
        dat_ra_kwargs: dict = None,
        n_vidat_layers: int = 2,
    ):
        """
        Assumes rgb input: B,T,C,H,W
        Assumes low_dim input: B,T,D

        fuse_mode:
            'modality-attention'  – ViT CLS token per frame, self-attention across modalities
            'bi-cross-attention'  – all ViT patch tokens, bidirectional cross-attention with force
        """
        super().__init__()

        rgb_keys = list()
        low_dim_keys = list()
        wrench_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        assert fuse_mode in ("modality-attention", "bi-cross-attention"), \
            f"Unknown fuse_mode: {fuse_mode}"

        # ── vision encoder ────────────────────────────────────────────────────
        vision_encoder = timm.create_model(
            model_name=vision_encoder_cfg.model_name,
            pretrained=vision_encoder_cfg.pretrained,
            global_pool=vision_encoder_cfg.global_pool,
            num_classes=0,
            img_size=224,
        )
        if vision_encoder_cfg.frozen:
            assert vision_encoder_cfg.pretrained
            for param in vision_encoder.parameters():
                param.requires_grad = False
        if vision_encoder_cfg.use_group_norm and not vision_encoder_cfg.pretrained:
            vision_encoder = replace_submodules(
                root_module=vision_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=(
                        (x.num_features // 16)
                        if (x.num_features % 16 == 0)
                        else (x.num_features // 8)
                    ),
                    num_channels=x.num_features,
                ),
            )
        self.v_feature_dim = 768  # ViT-base

        # ── force encoder ─────────────────────────────────────────────────────
        force_encoder = ForceEncoder(force_encoder_cfg.feature_dim)
        if force_encoder_cfg.frozen:
            for param in force_encoder.parameters():
                param.requires_grad = False

        # ── parse shape_meta ──────────────────────────────────────────────────
        image_shape = None
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            if attr.get("type", "low_dim") == "rgb":
                assert image_shape is None or image_shape == shape[1:]
                image_shape = shape[1:]

        # vision transforms
        if vision_encoder_cfg.transforms is not None and not isinstance(
            vision_encoder_cfg.transforms[0], torch.nn.Module
        ):
            assert vision_encoder_cfg.transforms[0].type == "RandomCrop"
            ratio = vision_encoder_cfg.transforms[0].ratio
            vision_encoder_cfg.transforms = [
                torchvision.transforms.RandomCrop(size=int(image_shape[0] * ratio)),
                torchvision.transforms.Resize(size=image_shape[0], antialias=True),
            ] + vision_encoder_cfg.transforms[1:]
        vision_transform = (
            nn.Identity()
            if vision_encoder_cfg.transforms is None
            else torch.nn.Sequential(*vision_encoder_cfg.transforms)
        )

        # assign keys → models
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            type = attr.get("type", "low_dim")
            key_shape_map[key] = shape
            if type == "rgb":
                rgb_keys.append(key)
                key_model_map[key] = (
                    vision_encoder
                    if vision_encoder_cfg.share_rgb_model
                    else copy.deepcopy(vision_encoder)
                )
                key_transform_map[key] = vision_transform
            elif type == "low_dim":
                if "wrench" in key:
                    wrench_keys.append(key)
                    key_model_map[key] = (
                        force_encoder
                        if force_encoder_cfg.share_force_model
                        else copy.deepcopy(force_encoder)
                    )
                else:
                    if not attr.get("ignore_by_policy", False):
                        low_dim_keys.append(key)
            elif type == "timestamp":
                pass
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)
        logger.info("rgb_keys:     %s", rgb_keys)
        logger.info("wrench_keys:  %s", wrench_keys)
        logger.info("low_dim_keys: %s", low_dim_keys)

        self.vision_encoder_cfg = vision_encoder_cfg
        self.force_encoder_cfg = force_encoder_cfg
        self.shape_meta = shape_meta
        self.fuse_mode = fuse_mode
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.wrench_keys = wrench_keys
        self.key_shape_map = key_shape_map
        self.position_encoding = position_encoding
        self.use_relational_features = use_relational_features
        self.symbol_retriever = symbol_retriever
        self.symbol_retriever_cfg = symbol_retriever_cfg
        self.bi_cross_heads = bi_cross_heads
        self.n_heads_sa = n_heads_sa
        self.n_heads_ra = n_heads_ra
        self.share_attn_params = share_attn_params

        rgb_horizon = shape_meta["sample"]["obs"]["sparse"]["rgb_0"]["horizon"]

        # ── fuse-mode specific modules ────────────────────────────────────────
        if fuse_mode == "modality-attention":
            # 1 CLS token per frame + 1 token per wrench key
            n_features = len(rgb_keys) * rgb_horizon + len(wrench_keys)
            self.transformer_encoder = torch.nn.TransformerEncoderLayer(
                d_model=self.v_feature_dim,
                nhead=8,
                dim_feedforward=2048,
                batch_first=True,
                dropout=0.0,
            )
            self.linear_projection = nn.Linear(
                self.v_feature_dim * n_features, self.v_feature_dim
            )
            if position_encoding == "learnable":
                self.position_embedding = torch.nn.Parameter(
                    torch.randn(n_features, self.v_feature_dim)
                )

        if self.fuse_mode == "bi-cross-attention":
            # all patch tokens + CLS per frame: (H/patch_size * W/patch_size) + 1
            n_patches = (
                (image_shape[0] // vision_encoder_cfg.downsample_ratio)
                * (image_shape[1] // vision_encoder_cfg.downsample_ratio)
            )
            tokens_per_frame = n_patches + 1
            total_tokens = (
                len(rgb_keys) * rgb_horizon * tokens_per_frame + len(wrench_keys)
            )
            self.img_cross_attention = CrossAttention(
                model_dim=self.v_feature_dim, num_heads=self.bi_cross_heads
            )
            self.force_cross_attention = CrossAttention(
                model_dim=self.v_feature_dim, num_heads=self.bi_cross_heads
            )
            self.attn_pool = AttentionPool1d(
                seq_len=total_tokens,
                embed_dim=self.v_feature_dim,
                num_heads=self.v_feature_dim // 64,
                output_dim=self.v_feature_dim,
            )
        # ViDAT: relational processing on visual patch tokens (per-frame, before fusion)
        if self.fuse_mode == "bi-cross-attention" and use_relational_features:
            symbol_dim = self.symbol_retriever_cfg.symbol_dim
            assert symbol_dim == self.v_feature_dim, (
                f"symbol_dim ({symbol_dim}) must equal v_feature_dim ({self.v_feature_dim}) "
                "because RelationalAttention projects symbols with d_model-sized weights."
            )
            # symbol retriever sized for per-frame token count only (not total_tokens)
            if symbol_retriever == "positional":
                self.symbol_retriever_module = PositionalSymbolRetriever(
                    symbol_dim=symbol_dim,
                    max_length=tokens_per_frame,
                )
            elif symbol_retriever == "position_relative":
                self.symbol_retriever_module = PositionRelativeSymbolRetriever(
                    symbol_dim=symbol_dim,
                    max_rel_pos=symbol_retriever_cfg.max_rel_pos,
                )
            elif symbol_retriever == "symbolic":
                self.symbol_retriever_module = SymbolicAttention(
                    d_model=self.v_feature_dim,
                    n_heads=symbol_retriever_cfg.n_heads,
                    n_symbols=symbol_retriever_cfg.n_symbols,
                )
            elif symbol_retriever == "relational_symbolic":
                self.symbol_retriever_module = RelationalSymbolicAttention(
                    d_model=self.v_feature_dim,
                    rel_n_heads=symbol_retriever_cfg.rel_n_heads,
                    symbolic_attn_n_heads=symbol_retriever_cfg.symbolic_attn_n_heads,
                    n_symbols=symbol_retriever_cfg.n_symbols,
                    nbhd_delta=symbol_retriever_cfg.nbhd_delta,
                )
            else:
                raise ValueError(f"Unknown symbol retriever type: {symbol_retriever}")

            # stacked ViDAT layers operating on per-frame patch tokens (B*T, n_patches, D)
            self.DAT_encoder = nn.ModuleList([
                DualAttnEncoderBlock(
                    d_model=self.v_feature_dim,
                    n_heads_sa=n_heads_sa,
                    n_heads_ra=n_heads_ra,
                    dff=dat_dff,
                    activation=dat_activation,
                    dropout_rate=dat_dropout_rate,
                    norm_first=dat_norm_first,
                    share_attn_params=share_attn_params,
                    ra_kwargs=dat_ra_kwargs,
                )
                for _ in range(n_vidat_layers)
            ])

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(self, obs_dict):
        """
        obs_dict values:
            rgb:     (B, T, C, H, W)
            wrench:  (B, T, D)
            low_dim: (B, T, D)
        """
        modality_features = []  # list of (B, seq_len, D)
        low_dim_features = []
        batch_size = next(iter(obs_dict.values())).shape[0]

        # ── rgb ───────────────────────────────────────────────────────────────
        for key in self.rgb_keys:
            img = obs_dict[key]  # (B, T, C, H, W)
            B, T = img.shape[:2]
            assert B == batch_size
            img = img.reshape(B * T, *img.shape[2:])
            img = self.key_transform_map[key](img)

            if self.vision_encoder_cfg.frozen:
                with torch.no_grad():
                    raw_feature = self.key_model_map[key](img)  # (B*T, 1+n_reg+n_patches, D)
            else:
                raw_feature = self.key_model_map[key](img)  # (B*T, 1+n_reg+n_patches, D)

            # strip register tokens if present (e.g. DINOv3): keep CLS + patch tokens only
            reg_token = getattr(self.key_model_map[key], 'reg_token', None)
            n_reg = reg_token.shape[1] if reg_token is not None else 0
            if n_reg > 0:
                raw_feature = torch.cat([raw_feature[:, :1], raw_feature[:, 1 + n_reg:]], dim=1)
            # → (B*T, 1+n_patches, D)

            if self.fuse_mode == "modality-attention":
                # CLS token only → (B, T, D)
                feature = raw_feature[:, 0, :]
                modality_features.append(feature.reshape(B, T, -1))
            else:
                # apply ViDAT per-frame before temporal reshape
                if self.use_relational_features:
                    symbols = self.symbol_retriever_module(raw_feature)  # (B*T, n_patches, symbol_dim)
                    for dat_layer in self.DAT_encoder:
                        raw_feature = dat_layer(raw_feature, symbols)    # (B*T, n_patches, D)
                # all tokens → (B, T*(L+1), D)
                feature = raw_feature.reshape(B, T * raw_feature.shape[1], -1)
                modality_features.append(feature)

        # ── wrench ────────────────────────────────────────────────────────────
        for key in self.wrench_keys:
            data = obs_dict[key]  # (B, T, 6)
            B, T = data.shape[:2]
            assert B == batch_size
            data = data.permute(0, 2, 1)  # (B, 6, T)
            feature = self.key_model_map[key](data.float())  # (B, D, T_out)
            feature = feature[:, :, -1]                      # (B, D) last causal step (original was first)
            modality_features.append(feature.unsqueeze(1))  # (B, 1, D)

        # ── low_dim ───────────────────────────────────────────────────────────
        for key in self.low_dim_keys:
            data = obs_dict[key]  # (B, T, D)
            B, T = data.shape[:2]
            assert B == batch_size
            low_dim_features.append(data.reshape(B, -1))

        # ── fusion ────────────────────────────────────────────────────────────
        if self.fuse_mode == "modality-attention":
            in_embeds = torch.cat(modality_features, dim=1)  # (B, n_features, D)
            if self.position_encoding == "learnable":
                if self.position_embedding.device != in_embeds.device:
                    self.position_embedding = self.position_embedding.to(in_embeds.device)
                in_embeds = in_embeds + self.position_embedding
            out_embeds = self.transformer_encoder(in_embeds)   # (B, n_features, D)
            result = self.linear_projection(out_embeds.reshape(B, -1))  # (B, D)

        elif self.fuse_mode == "bi-cross-attention" and not self.use_relational_features:
            n_rgb = len(self.rgb_keys)
            rgb_tokens   = torch.cat(modality_features[:n_rgb], dim=1)  # (B, T*(L+1), D)
            force_tokens = torch.cat(modality_features[n_rgb:], dim=1)  # (B, n_wrench, D)

            img_enhanced   = self.img_cross_attention(rgb_tokens, force_tokens)    # (B, T*(L+1), D)
            force_enhanced = self.force_cross_attention(force_tokens, rgb_tokens)  # (B, n_wrench, D)

            fused  = torch.cat([img_enhanced, force_enhanced], dim=1)  # (B, total_tokens, D)
            result = self.attn_pool(fused)                              # (B, D)

        elif self.fuse_mode == "bi-cross-attention" and self.use_relational_features:
            n_rgb = len(self.rgb_keys)
            rgb_tokens   = torch.cat(modality_features[:n_rgb], dim=1)  # (B, T*(L+1), D)
            force_tokens = torch.cat(modality_features[n_rgb:], dim=1)  # (B, n_wrench, D)

            img_enhanced   = self.img_cross_attention(rgb_tokens, force_tokens)    # (B, T*(L+1), D)
            force_enhanced = self.force_cross_attention(force_tokens, rgb_tokens)  # (B, n_wrench, D)

            fused  = torch.cat([img_enhanced, force_enhanced], dim=1)  # (B, total_tokens, D)
            result = self.attn_pool(fused)                              # (B, D)

        if low_dim_features:
            result = torch.cat([result, torch.cat(low_dim_features, dim=-1)], dim=-1)

        return result

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta["obs"]
        sample_obs_shape_meta = self.shape_meta["sample"]["obs"]["sparse"]
        for key, attr in obs_shape_meta.items():
            if attr.get("type", "low_dim") == "timestamp":
                continue
            shape = tuple(attr["shape"])
            horizon = sample_obs_shape_meta[key]["horizon"]
            example_obs_dict[key] = torch.zeros(
                (1, horizon) + shape, dtype=self.dtype, device=self.device
            )
        example_output = self.forward(example_obs_dict)
        assert len(example_output.shape) == 2
        assert example_output.shape[0] == 1
        return example_output.shape

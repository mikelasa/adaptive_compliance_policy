import copy
import numpy as np

import timm
import torch
import torch.nn as nn
import torchvision
import logging

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import replace_submodules
from diffusion_policy.model.vision.utils.attention_viz import (
    VizTransformerEncoderLayer,
    VizCrossAttention,
)
import diffusion_policy.model.vision.utils.attention_viz as attention_viz

from multimodal_representation.multimodal.models.base_models.encoders import (
    ForceEncoder,
)
from diffusion_policy.model.vision.ft_embed import FTEmbed

from dual_attention.dual_attn_blocks import DualAttnEncoderBlock
from dual_attention.symbol_retrieval import (
    PositionalSymbolRetriever,
    PositionRelativeSymbolRetriever,
    SymbolicAttention,
    RelationalSymbolicAttention,
)

logger = logging.getLogger(__name__)


class TimmObsEncoderBiCrossDATTransformer(ModuleAttrMixin):
    def __init__(
        self,
        shape_meta: dict,
        fuse_mode: str,
        reduce_pretrained_lr: bool,
        vision_encoder_cfg: dict,
        force_encoder_cfg: dict,
        second_camera: bool = False,
        position_encoding: str = "learnable",
        use_relational_features: bool = False,
        symbol_retriever: str = "positional",
        symbol_retriever_cfg: dict = None,
        bi_cross_heads: int = 4,
        bi_cross_attn_drop: float = 0.0,
        bi_cross_drop: float = 0.0,
        n_heads_sa: int = 4,
        n_heads_ra: int = 4,
        share_attn_params: bool = False,
        dat_dff: int = 2048,
        dat_activation: str = "relu",
        dat_dropout_rate: float = 0.0,
        dat_norm_first: bool = True,
        dat_ra_kwargs: dict = None,
        dat_n_layers: int = 1,
    ):
        """
        Token-sequence variant of TimmObsEncoderWithForceV1 for transformer diffusion
        policies. Returns (B, N, D) instead of (B, D) — AttentionPool1d is removed so
        the full fused sequence is passed directly to the transformer denoiser.

        fuse_mode:
            'modality-attention'     – CLS token per frame + self-attn across modalities;
                                       returns (B, n_features [+ n_lowdim], D)
            'bi-cross-attention'     – all patch tokens, bidirectional cross-attn with force;
                                       returns (B, total_tokens [+ n_lowdim], D)
            'bi-cross-attention-DAT' – same + DAT relational encoder;
                                       returns (B, total_tokens [+ n_lowdim], D)
            'DAT'                    – all tokens concatenated + DAT encoder;
                                       returns (B, total_tokens [+ n_lowdim], D)

        low_dim keys (non-wrench) are projected to v_feature_dim and appended as tokens.
        """
        super().__init__()

        rgb_keys = list()
        low_dim_keys = list()
        wrench_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        assert fuse_mode in ("modality-attention", "bi-cross-attention", "bi-cross-attention-DAT", "DAT"), \
            f"Unknown fuse_mode: {fuse_mode}"

        # ── vision encoder ────────────────────────────────────────────────────
        vision_encoder = timm.create_model(
            model_name=vision_encoder_cfg.model_name,
            pretrained=vision_encoder_cfg.pretrained,
            global_pool=vision_encoder_cfg.global_pool,
            num_classes=0,
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
        self.v_feature_dim = vision_encoder.embed_dim

        # ── force encoder ─────────────────────────────────────────────────────
        force_encoder_type = getattr(force_encoder_cfg, "type", "fft")
        if force_encoder_type == "fft":
            force_encoder = ForceEncoder(force_encoder_cfg.feature_dim)
        elif force_encoder_type == "cnn1d":
            force_encoder = FTEmbed(
                ft_dim=getattr(force_encoder_cfg, "ft_dim", 6),
                hidden_channels=force_encoder_cfg.feature_dim,
                norm_type=getattr(force_encoder_cfg, "norm_type", "group"),
                act=getattr(force_encoder_cfg, "act", "gelu"),
                alpha_init=getattr(force_encoder_cfg, "alpha_init", 1e-2),
            )
        else:
            raise ValueError(f"Unknown force_encoder type: {force_encoder_type!r}")
        self.force_encoder_type = force_encoder_type
        self.fft_last_n_tokens = 1
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
                if rgb_keys and not second_camera:
                    continue
                if vision_encoder_cfg.share_rgb_model:
                    key_model_map[key] = vision_encoder
                elif not rgb_keys:
                    key_model_map[key] = copy.deepcopy(vision_encoder)
                else:
                    key_model_map[key] = copy.deepcopy(key_model_map[rgb_keys[0]])
                rgb_keys.append(key)
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

        # ── low_dim projection to token space ─────────────────────────────────
        # Each low_dim key (B, T, D_k) is projected to (B, T, v_feature_dim) tokens.
        key_lowdim_proj_map = nn.ModuleDict()
        for key in low_dim_keys:
            d_k = int(np.prod(key_shape_map[key]))
            key_lowdim_proj_map[key] = (
                nn.Identity() if d_k == self.v_feature_dim
                else nn.Linear(d_k, self.v_feature_dim)
            )
        self.key_lowdim_proj_map = key_lowdim_proj_map

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

        rgb_horizon = shape_meta["sample"]["obs"]["sparse"][rgb_keys[0]]["horizon"]

        # ── fuse-mode specific modules ────────────────────────────────────────
        if fuse_mode == "modality-attention":
            assert force_encoder_type != "cnn1d", (
                "fuse_mode='modality-attention' is incompatible with force_encoder_cfg.type='cnn1d'."
            )
            n_features = len(rgb_keys) * rgb_horizon + len(wrench_keys)
            self.transformer_encoder = VizTransformerEncoderLayer(
                d_model=self.v_feature_dim,
                nhead=8,
                dim_feedforward=2048,
                batch_first=True,
                dropout=0.0,
            )
            # No linear_projection: the full (B, n_features, D) token sequence is returned.
            if position_encoding == "learnable":
                self.position_embedding = torch.nn.Parameter(
                    torch.randn(n_features, self.v_feature_dim)
                )

        if self.fuse_mode in ("bi-cross-attention", "bi-cross-attention-DAT", "DAT"):
            n_patches = (
                (image_shape[0] // vision_encoder_cfg.downsample_ratio)
                * (image_shape[1] // vision_encoder_cfg.downsample_ratio)
            )
            tokens_per_frame = n_patches + 1
            fft_last_n_tokens = int(getattr(force_encoder_cfg, "last_n_tokens", 1))
            if force_encoder_type == "fft" and wrench_keys:
                _wh = shape_meta["sample"]["obs"]["sparse"][wrench_keys[0]]["horizon"]
                with torch.no_grad():
                    _dummy_out = force_encoder(torch.zeros(1, 6, _wh))
                _actual_t_out = _dummy_out.shape[-1]
                if fft_last_n_tokens > _actual_t_out:
                    logger.warning(
                        "fft_last_n_tokens=%d > ForceEncoder output length=%d "
                        "(wrench_horizon=%d); clamping to %d.",
                        fft_last_n_tokens, _actual_t_out, _wh, _actual_t_out,
                    )
                    fft_last_n_tokens = _actual_t_out
            self.fft_last_n_tokens = fft_last_n_tokens
            if force_encoder_type == "cnn1d" and wrench_keys:
                wrench_horizon = shape_meta["sample"]["obs"]["sparse"][wrench_keys[0]]["horizon"]
                n_force_tokens = len(wrench_keys) * wrench_horizon
            else:
                n_force_tokens = len(wrench_keys) * fft_last_n_tokens

            if self.fuse_mode in ("bi-cross-attention", "bi-cross-attention-DAT"):
                self.img_cross_attention = VizCrossAttention(
                    model_dim=self.v_feature_dim, num_heads=self.bi_cross_heads,
                    attn_drop=bi_cross_attn_drop, drop=bi_cross_drop,
                )
                self.force_cross_attention = VizCrossAttention(
                    model_dim=self.v_feature_dim, num_heads=self.bi_cross_heads,
                    attn_drop=bi_cross_attn_drop, drop=bi_cross_drop,
                )

            # total_tokens is the fused sequence length before low_dim tokens are appended
            total_tokens = len(rgb_keys) * rgb_horizon * tokens_per_frame + n_force_tokens
            self.total_tokens = total_tokens

        if self.fuse_mode in ("bi-cross-attention-DAT", "DAT"):
            symbol_dim = self.symbol_retriever_cfg.symbol_dim
            assert symbol_dim == self.v_feature_dim, (
                f"symbol_dim ({symbol_dim}) must equal v_feature_dim ({self.v_feature_dim})."
            )
            if symbol_retriever == "positional":
                self.symbol_retriever_module = PositionalSymbolRetriever(
                    symbol_dim=symbol_dim,
                    max_length=total_tokens,
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
                for _ in range(dat_n_layers)
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

        Returns:
            tokens: (B, N, v_feature_dim)
        """
        rgb_features = []
        force_features = []
        low_dim_tokens = []
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
                    raw_feature = self.key_model_map[key](img)
            else:
                raw_feature = self.key_model_map[key](img)

            reg_token = getattr(self.key_model_map[key], 'reg_token', None)
            n_reg = reg_token.shape[1] if reg_token is not None else 0
            if n_reg > 0:
                raw_feature = torch.cat([raw_feature[:, :1], raw_feature[:, 1 + n_reg:]], dim=1)

            if self.fuse_mode == "modality-attention":
                feature = raw_feature[:, 0, :]              # CLS token → (B*T, D)
                rgb_features.append(feature.reshape(B, T, -1))
            else:
                feature = raw_feature.reshape(B, T * raw_feature.shape[1], -1)  # (B, T*(L+1), D)
                rgb_features.append(feature)

        rgb_tokens = torch.cat(rgb_features, dim=1)

        # ── wrench ────────────────────────────────────────────────────────────
        for key in self.wrench_keys:
            data = obs_dict[key]  # (B, T, 6)
            B, T = data.shape[:2]
            assert B == batch_size
            if self.force_encoder_type == "fft":
                data = data.permute(0, 2, 1)
                feature = self.key_model_map[key](data.float())
                feature = feature[:, :, -self.fft_last_n_tokens:].permute(0, 2, 1)  # (B, last_n, D)
                force_features.append(feature)
            else:
                feature = self.key_model_map[key](data.float())  # (B, T, D)
                force_features.append(feature)

        # ── low_dim → tokens ──────────────────────────────────────────────────
        for key in self.low_dim_keys:
            data = obs_dict[key]  # (B, T, D_k)
            B, T = data.shape[:2]
            assert B == batch_size
            proj = self.key_lowdim_proj_map[key]
            tokens = proj(data.reshape(B, T, -1).float())  # (B, T, v_feature_dim)
            low_dim_tokens.append(tokens)

        if low_dim_tokens:
            low_dim_tokens = [torch.cat(low_dim_tokens, dim=1)]  # (B, n_lowdim, D)

        # ── fusion ────────────────────────────────────────────────────────────
        if self.fuse_mode == "modality-attention":
            force_tokens = torch.cat(force_features, dim=1) if force_features else None
            in_embeds = torch.cat([rgb_tokens, force_tokens], dim=1) if force_tokens is not None else rgb_tokens
            if self.position_encoding == "learnable":
                if self.position_embedding.device != in_embeds.device:
                    self.position_embedding = self.position_embedding.to(in_embeds.device)
                in_embeds = in_embeds + self.position_embedding
            result = self.transformer_encoder(in_embeds)  # (B, n_features, D)

            # attention visualization (inference only; no-op when disabled)
            if hasattr(self.transformer_encoder, "last_attn_weights"):
                self._captured_fusion_attn = {
                    "kind": "self",
                    "self_attn_per_key": self.transformer_encoder.last_attn_weights[0]
                        .float().mean(dim=0).cpu().numpy(),  # (L,) avg over queries
                    "n_img": int(rgb_tokens.shape[1]),
                }
                del self.transformer_encoder.last_attn_weights

        elif self.fuse_mode == "bi-cross-attention":
            force_tokens = torch.cat(force_features, dim=1)
            img_enhanced   = self.img_cross_attention(rgb_tokens, force_tokens)
            force_enhanced = self.force_cross_attention(force_tokens, rgb_tokens)
            result = torch.cat([img_enhanced, force_enhanced], dim=1)  # (B, total_tokens, D)
            self._capture_cross_fusion_attn(rgb_tokens, force_tokens)

        elif self.fuse_mode == "bi-cross-attention-DAT":
            force_tokens = torch.cat(force_features, dim=1)
            img_enhanced   = self.img_cross_attention(rgb_tokens, force_tokens)
            force_enhanced = self.force_cross_attention(force_tokens, rgb_tokens)
            fused = torch.cat([img_enhanced, force_enhanced], dim=1)
            for layer in self.DAT_encoder:
                symbols = self.symbol_retriever_module(fused)
                fused   = layer(fused, symbols)
            result = fused  # (B, total_tokens, D)
            self._capture_cross_fusion_attn(rgb_tokens, force_tokens)

        elif self.fuse_mode == "DAT":
            force_tokens = torch.cat(force_features, dim=1)
            fused = torch.cat([rgb_tokens, force_tokens], dim=1)
            for layer in self.DAT_encoder:
                symbols = self.symbol_retriever_module(fused)
                fused   = layer(fused, symbols)
            result = fused  # (B, total_tokens, D)
            # no cross-attention in DAT-only mode, so there's no VizCrossAttention
            # to hasattr-check against — gate directly on the module flag instead.
            # Still record token counts so the denoiser-side capture can be split
            # into img/force segments even though the encoder has nothing of its
            # own to show.
            if attention_viz.VISUALIZE_ATTENTION:
                self._captured_fusion_attn = {
                    "kind": "none",
                    "n_img": int(rgb_tokens.shape[1]),
                    "n_force": int(force_tokens.shape[1]),
                }

        # append low_dim tokens along the sequence dimension
        if low_dim_tokens:
            result = torch.cat([result] + low_dim_tokens, dim=1)

        return result  # (B, N, D)

    def _capture_cross_fusion_attn(self, rgb_tokens, force_tokens):
        """Stash bi-cross-attention weights (img<-force and force<-img) captured
        by VizCrossAttention into self._captured_fusion_attn, for the policy to
        pull via pop_fusion_attention_viz() and pickle alongside the denoiser's
        attention. No-op (leaves _captured_fusion_attn unset) when capture is off.
        """
        cross_img_w   = getattr(self.img_cross_attention,   "last_attn_weights", None)
        cross_force_w = getattr(self.force_cross_attention, "last_attn_weights", None)
        if cross_img_w is None and cross_force_w is None:
            return
        capture = {
            "kind": "cross",
            "n_img": int(rgb_tokens.shape[1]),
            "n_force": int(force_tokens.shape[1]),
        }
        if cross_img_w is not None:
            # avg over image queries -> (n_force,): which force timesteps img patches use
            capture["cross_per_force_step"] = cross_img_w[0].float().cpu().numpy().mean(axis=0)
            del self.img_cross_attention.last_attn_weights
        if cross_force_w is not None:
            # avg over force queries -> (n_img,): which image patches force tokens use
            capture["cross_per_img_token"] = cross_force_w[0].float().cpu().numpy().mean(axis=0)
            del self.force_cross_attention.last_attn_weights
        self._captured_fusion_attn = capture

    def pop_fusion_attention_viz(self):
        """Return and clear the fusion attention captured during the last forward()
        call (None if nothing was captured, i.e. VISUALIZE_ATTENTION was off)."""
        capture = getattr(self, "_captured_fusion_attn", None)
        self._captured_fusion_attn = None
        return capture

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
        assert len(example_output.shape) == 3, (
            f"Expected 3D output (B, N, D), got shape {example_output.shape}"
        )
        assert example_output.shape[0] == 1
        return example_output.shape  # (1, N, D)

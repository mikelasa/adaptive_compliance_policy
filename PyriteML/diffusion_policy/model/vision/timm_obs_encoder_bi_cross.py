import copy

import timm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import logging

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import replace_submodules
from diffusion_policy.model.vision.utils.cross_attention import CrossAttention
from diffusion_policy.model.vision.utils.attention_pool import AttentionPool1d

# from diffusion_policy.model.vision.force_spec_encoder import ForceSpecEncoder, convert_to_spec
from multimodal_representation.multimodal.models.base_models.encoders import (
    ForceEncoder,
    ProprioEncoder,
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
        # replace BatchNorm with GroupNorm
        # use single rgb model for all rgb inputs
        # renormalize rgb input with imagenet normalization
        # assuming input in [0,1]
        position_encoding: str = "learnable",
    ):
        """
        Assumes rgb input: B,T,C,H,W
        Assumes low_dim input: B,T,D
        """

        super().__init__()

        # parse shapes
        rgb_keys = list()
        low_dim_keys = list()
        wrench_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        model_list = []
        feature_dim_list = []

        # create config for vision encoder and force encoder
        for cfg in [vision_encoder_cfg, force_encoder_cfg]:
            # force encoder is 5 layer 1D conv with stride 2, input 6 dim force data, output feature dim 768
            if cfg.model_name == "causalconv":
                model = ForceEncoder(cfg.feature_dim)
            else:
                model = timm.create_model(
                    model_name=cfg.model_name,
                    pretrained=cfg.pretrained,
                    global_pool=cfg.global_pool,
                    num_classes=0,
                )
            
            # if frozen, no gradients
            if cfg.frozen:
                assert cfg.pretrained
                for param in model.parameters():
                    param.requires_grad = False
            
            #  feature dim setup
            if cfg.model_name == "causalconv":
                feature_dim = cfg.feature_dim
            else:
                feature_dim = 768
            #append to list
            feature_dim_list.append(feature_dim)

            # add group norm if specified
            if cfg.use_group_norm and not cfg.pretrained:
                model = replace_submodules(
                    root_module=model,
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
            model_list.append(model)
            
        #model list
        vision_encoder, force_encoder = model_list
        self.v_feature_dim, self.f_feature_dim = feature_dim_list

        # assign keys to encoders based on input shape
        image_shape = None
        obs_shape_meta = shape_meta["obs"]
        # meta_shape has rgb and low_dim keys
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            type = attr.get("type", "low_dim")
            if type == "rgb":
                assert image_shape is None or image_shape == shape[1:]
                image_shape = shape[1:]
        # apply transforms if specified in config, for vision encoder only
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

        # assign keys to encoders and transforms
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            type = attr.get("type", "low_dim")
            key_shape_map[key] = shape
            # for rgb input, assign vision encoder and vision transform
            if type == "rgb":
                rgb_keys.append(key)

                vision_encoder = (
                    vision_encoder
                    if vision_encoder_cfg.share_rgb_model
                    else copy.deepcopy(vision_encoder)
                )
                key_model_map[key] = vision_encoder
                key_transform_map[key] = vision_transform
            # for low_dim input, assign force encoder if it's wrench data, otherwise ignore by default (can be included by setting "ignore_by_policy" to False in config)
            elif type == "low_dim":
                if "wrench" in key:
                    print("key_model_map adding wrench key:", key)
                    wrench_keys.append(key)

                    force_encoder = (
                        force_encoder
                        if force_encoder_cfg.share_force_model
                        else copy.deepcopy(force_encoder)
                    )
                    key_model_map[key] = force_encoder
                else:
                    if not attr.get("ignore_by_policy", False):
                        low_dim_keys.append(key)
            elif type == "timestamp":
                pass
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
            
        # 
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)
        print("rgb keys:         ", rgb_keys)
        print("low_dim_keys keys:", low_dim_keys)

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

        if vision_encoder_cfg.model_name.startswith("vit"):
            # if vision encoder is vit
            if vision_encoder_cfg.feature_aggregation == "all_tokens":
                # Use all tokens from ViT
                pass
            elif vision_encoder_cfg.feature_aggregation is not None:
                logger.warn(
                    f"vit will use the CLS token. feature_aggregation ({vision_encoder_cfg.feature_aggregation}) is ignored!"
                )
                vision_encoder_cfg.feature_aggregation = None
        # TODO: add dinoV3


        #fuse mode is self attention
        if fuse_mode == "modality-attention":
            self.transformer_encoder = torch.nn.TransformerEncoderLayer(
                d_model=self.v_feature_dim,
                nhead=8,
                dim_feedforward=2048,
                batch_first=True,
                dropout=0.0,
            )
            rgb_horizon = shape_meta["sample"]["obs"]["sparse"]["rgb_0"]["horizon"]
            if vision_encoder_cfg.feature_aggregation == "all_tokens":
                # each ViT frame produces (H/patch * W/patch + 1) tokens (patches + CLS)
                n_patches = (image_shape[0] // vision_encoder_cfg.downsample_ratio) * \
                            (image_shape[1] // vision_encoder_cfg.downsample_ratio)
                n_img_tokens_per_frame = n_patches + 1  # +1 for CLS
                n_rgb_tokens = len(rgb_keys) * rgb_horizon * n_img_tokens_per_frame
            else:
                # CLS only: 1 token per frame
                n_rgb_tokens = len(rgb_keys) * rgb_horizon
            n_features = n_rgb_tokens + len(wrench_keys)
            self.linear_projection = nn.Linear(
                self.v_feature_dim * n_features, self.v_feature_dim
            )
            if position_encoding == "learnable":
                self.position_embedding = torch.nn.Parameter(
                    torch.randn(n_features, self.v_feature_dim)
                )

                # integrate bi directional cross attention between modalities
        elif fuse_mode == "bi-cross-attention":
            self.img_cross_attention   = CrossAttention(model_dim=self.v_feature_dim, num_heads=4)
            self.force_cross_attention = CrossAttention(model_dim=self.v_feature_dim, num_heads=4)

            # compute fixed token count for the pool
            rgb_horizon = shape_meta["sample"]["obs"]["sparse"][rgb_keys[0]]["horizon"]
            if vision_encoder_cfg.feature_aggregation == "all_tokens":
                n_patches = (image_shape[0] // vision_encoder_cfg.downsample_ratio) * \
                            (image_shape[1] // vision_encoder_cfg.downsample_ratio)
                tokens_per_frame = n_patches + 1  # +1 for CLS
            else:
                tokens_per_frame = 1  # CLS only
            total_tokens = len(rgb_keys) * rgb_horizon * tokens_per_frame + len(wrench_keys)

            self.attn_pool = AttentionPool1d(
                seq_len=total_tokens,
                embed_dim=self.v_feature_dim,
                num_heads=8,
                output_dim=self.v_feature_dim
            )

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def aggregate_feature(self, model_name, agg_mode, feature):
        if model_name.startswith("vit"):
            if agg_mode is None:
                # Use only CLS token
                return feature[:, 0, :]  # (B, D)
            elif agg_mode == "all_tokens":
                # Use all tokens (including CLS, or skip CLS if desired)
                return feature  # (B, N+1, D)
            else:
                raise ValueError(f"Unknown agg_mode: {agg_mode}")
        

    def forward(self, obs_dict):
        """Assume each image key is (B, T, C, H, W)"""
        features = list()
        modality_features = list()
        low_dim_features = list()
        batch_size = next(iter(obs_dict.values())).shape[0]

        for key in self.rgb_keys:
            img = obs_dict[key]  # (B, T, 3, H, W)
            B, T = img.shape[:2]
            assert B == batch_size
            assert img.shape[2:] == self.key_shape_map[key]
            img = img.reshape(B * T, *img.shape[2:])  # (B*T, 3, H, W)
            img = self.key_transform_map[key](img)

            # frozen backbone: no activation storage for backprop
            if self.vision_encoder_cfg.frozen:
                with torch.no_grad():
                    raw_feature = self.key_model_map[key](img)
            else:
                raw_feature = self.key_model_map[key](img)

            feature = self.aggregate_feature(
                model_name=self.vision_encoder_cfg.model_name,
                agg_mode=self.vision_encoder_cfg.feature_aggregation,
                feature=raw_feature,
            )

            # depending on CLS or all tokens, feature shape is (B*T, D) or (B*T, N+1, D)
            if feature.ndim == 2:  # (B*T, D) - CLS only
                features.append(feature.reshape(B, -1))  # (B, T*D)
                modality_features.append(feature.reshape(B, T, -1))  # (B, T, D)
            elif feature.ndim == 3:  # (B*T, N, D) - all tokens
                N = feature.shape[1]
                feature = feature.reshape(B, T, N, -1)  # (B, T, N, D)
                feature_flat = feature.reshape(B, T * N, -1)  # (B, T*N, D)
                features.append(feature_flat.reshape(B, -1))  # (B, T*N*D)
                modality_features.append(feature_flat)  # (B, T*N, D)
            else:
                raise RuntimeError(f"Unexpected feature shape: {feature.shape}")

        for key in self.wrench_keys:
            data = obs_dict[key] # (B, T, 6)
            B, T = data.shape[:2]
            assert B == batch_size
            assert data.shape[2:] == self.key_shape_map[key]
            data = data.permute(0, 2, 1) # (B, 6, T) because ForceEncoder expects (B, C, T)
            feature = self.key_model_map[key](data.float()) #encoder (B, 768, T/32) 32 caused by 5 conv layers with stride 2, then take the last time step (B, 768)     
            feature = feature[:, :, -1] # take the last time step feature, which has the most recent force information (B, 768)
            assert len(feature.shape) == 2 and feature.shape[0] == B
            features.append(feature.reshape(B, -1)) # (B, 768)
            modality_features.append(feature.unsqueeze(1)) # (B, 1, 768)

        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            B, T = data.shape[:2]
            assert B == batch_size
            assert data.shape[2:] == self.key_shape_map[key]
            # directly concatenate actions
            features.append(data.reshape(B, -1))
            low_dim_features.append(data.reshape(B, -1))

        if self.fuse_mode == "modality-attention":
            #print RGB and force feature shapes before attention
            #print(f"modality features shapes before attention: {[f.shape for f in modality_features]}") # should be [(B, T, 768), (B, 1, 768)]
            in_embeds = torch.cat(modality_features, dim=1)  # [batch, n_features, D]
            #print(f"in_embeds shape: {in_embeds.shape}") # should be (B, T+1, 768) for CLS or (B, T*N+1, 768) for all tokens
            if self.position_encoding == "learnable":
                if self.position_embedding.device != in_embeds.device:
                    self.position_embedding = self.position_embedding.to(feature.device)
                in_embeds = in_embeds + self.position_embedding
            out_embeds = self.transformer_encoder(in_embeds)  # [batch, n_features, D]
            result = torch.concat(
                [out_embeds[:, i] for i in range(out_embeds.shape[1])], dim=1
            )
            result = self.linear_projection(result)
            result = torch.concat([result, torch.cat(low_dim_features, dim=-1)], dim=1)
        

        elif self.fuse_mode == "bi-cross-attention":
            n_rgb = len(self.rgb_keys)
            rgb_tokens   = torch.cat(modality_features[:n_rgb], dim=1)  # (B, T*N, 768)
            force_tokens = torch.cat(modality_features[n_rgb:], dim=1)  # (B, 1,   768)

            img_feat   = self.img_cross_attention(img_feat=rgb_tokens,   ft_feat=force_tokens)
            force_feat = self.force_cross_attention(img_feat=force_tokens, ft_feat=rgb_tokens)

            fused = torch.cat([img_feat, force_feat], dim=1)  # (B, total_tokens, 768)
            result = self.attn_pool(fused)                     # (B, 768)

            if low_dim_features:
                result = torch.cat([result, torch.cat(low_dim_features, dim=-1)], dim=-1)  # (B, 768 + 27)

        return result
    
    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta["obs"]
        sample_obs_shape_meta = self.shape_meta["sample"]["obs"]["sparse"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            if type == "timestamp":
                continue

            shape = tuple(attr["shape"])
            horizon = sample_obs_shape_meta[key]["horizon"]
            this_obs = torch.zeros(
                (1, horizon) + shape, dtype=self.dtype, device=self.device
            )
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        if self.fuse_mode == "bi-cross-attention":
            assert len(example_output.shape) == 2  # (1, total_tokens, D)
        else:
            assert len(example_output.shape) == 2  # (1, flat_dim)
        assert example_output.shape[0] == 1

        return example_output.shape


if __name__ == "__main__":
    timm_obs_encoder_with_force = TimmObsEncoderWithForceV2(
        shape_meta=None,
        model_name="resnet18.a1_in1k",
        pretrained=False,
        global_pool="",
        transforms=None,
    )

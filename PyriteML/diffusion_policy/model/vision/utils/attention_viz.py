"""
Attention visualization utilities for the observation encoder.

Mirrors the design of ImplicitRDP's transformer_utils.py: a module-level flag
toggles weight capture so training and inference share the exact same code path
with ZERO overhead when disabled.

IMPORTANT (encoder-specific caveat):
    torch.nn.TransformerEncoderLayer has a "sparsity fast path"
    (torch._transformer_encoder_layer_fwd) that, at eval time, bypasses
    _sa_block entirely. Overriding _sa_block alone is therefore NOT enough for
    the encoder (it is enough for the decoder used in ImplicitRDP, which has no
    such fast path). We also override forward() to force the slow path whenever
    capture is enabled, so _sa_block actually runs.

Usage (inference / eval script only):
    import diffusion_policy.model.vision.utils.attention_viz as av
    av.VISUALIZE_ATTENTION = True
    ...run predict_action()...
"""
from typing import Optional
import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn

from .cross_attention import CrossAttention
from .attention_pool import AttentionPool1d

# Flip to True ONLY in inference/eval scripts. Leave False for training.
VISUALIZE_ATTENTION = True


class VizTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """Drop-in replacement for nn.TransformerEncoderLayer that captures the
    averaged self-attention weight matrix into ``self.last_attn_weights``
    whenever the module-level VISUALIZE_ATTENTION flag is True.

    When the flag is False, forward() defers to the parent implementation,
    keeping the fast path and incurring no overhead.
    """

    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor],
                  key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        if VISUALIZE_ATTENTION:
            x, weights = self.self_attn(
                x, x, x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=True,  # average over heads → (B, L_q, L_k)
                is_causal=is_causal,
            )
            self.last_attn_weights = weights.detach()
        else:
            x = self.self_attn(
                x, x, x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
                is_causal=is_causal,
            )[0]
        return self.dropout1(x)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                is_causal: bool = False) -> Tensor:
        # Disabled: defer to parent (keeps the fast path → zero overhead).
        if not VISUALIZE_ATTENTION:
            return super().forward(src, src_mask, src_key_padding_mask, is_causal)

        # Enabled: force the slow path so the overridden _sa_block runs.
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x


class VizCrossAttention(CrossAttention):
    """Drop-in for CrossAttention (img←force) that captures attention weights
    into ``self.last_attn_weights`` when VISUALIZE_ATTENTION is True.

    Also fixes the parent's inefficiency of computing weights even when unused.
    weights shape when captured: (B, n_img_tokens, n_force_tokens), averaged over heads.
    """

    def forward(self, img_feat, ft_feat, img_key_padding_mask=None, ft_key_padding_mask=None):
        attn_img, weights = self.attn(
            query=img_feat,
            key=ft_feat,
            value=ft_feat,
            key_padding_mask=ft_key_padding_mask,
            need_weights=VISUALIZE_ATTENTION,
            average_attn_weights=True,
        )
        if VISUALIZE_ATTENTION:
            self.last_attn_weights = weights.detach()  # (B, n_img, n_force)
        img_out = self.norm1(img_feat + self.drop(attn_img))
        if self.use_mlp:
            img_out = self.norm2(img_out + self.drop(self.ffn(img_out)))
        return img_out


class VizTransformerDecoderLayer(nn.TransformerDecoderLayer):
    """Drop-in replacement for nn.TransformerDecoderLayer that captures the
    averaged cross-attention weight matrix (tgt queries -> memory keys) into
    ``self.last_attn_weights`` whenever VISUALIZE_ATTENTION is True.

    Unlike VizTransformerEncoderLayer, nn.TransformerDecoderLayer.forward()
    calls _mha_block directly with no sparsity fast-path bypassing it, so
    overriding _mha_block alone is sufficient — no need to override forward().
    """

    def _mha_block(self, x: Tensor, mem: Tensor,
                    attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor],
                    is_causal: bool = False) -> Tensor:
        if VISUALIZE_ATTENTION:
            x, weights = self.multihead_attn(
                x, mem, mem,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=True,  # average over heads -> (B, T_tgt, T_mem)
                is_causal=is_causal,
            )
            self.last_attn_weights = weights.detach()
        else:
            x = self.multihead_attn(
                x, mem, mem,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
                is_causal=is_causal,
            )[0]
        return self.dropout2(x)


class VizAttentionPool1d(AttentionPool1d):
    """Drop-in for AttentionPool1d that captures pooling weights into
    ``self.last_attn_weights`` when VISUALIZE_ATTENTION is True.

    weights shape when captured: (B, 1, L+1) where L is the input sequence length
    and the +1 is the prepended mean token at index 0.
    Token layout in keys: [mean(index 0), img_0..img_{n_img-1}, force_0..force_{n_force-1}]
    So image mass = weights[:, :, 1:1+n_img].sum() and force mass = weights[:, :, 1+n_img:].sum()
    """

    def forward(self, x):
        if not VISUALIZE_ATTENTION:
            return super().forward(x)

        # Replicate parent logic but with need_weights=True to capture weights.
        B, L, D = x.shape
        x_seq = x.permute(1, 0, 2)                                              # (L, B, D)
        x_seq = torch.cat([x_seq.mean(dim=0, keepdim=True), x_seq], dim=0)     # (L+1, B, D)
        x_seq = x_seq + self.positional_embedding[:, None, :].to(x_seq.dtype)

        x_out, weights = F.multi_head_attention_forward(
            query=x_seq[:1], key=x_seq, value=x_seq,
            embed_dim_to_check=x_seq.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=True,
            average_attn_weights=True,
        )
        self.last_attn_weights = weights.detach()  # (B, 1, L+1)
        return self.norm(x_out.squeeze(0))
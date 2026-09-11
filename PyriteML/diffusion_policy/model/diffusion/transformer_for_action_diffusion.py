from typing import Union, Optional, Tuple
import logging
import torch
import torch.nn as nn
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.vision.utils.attention_viz import VizTransformerDecoderLayer

logger = logging.getLogger(__name__)

class TransformerForActionDiffusion(ModuleAttrMixin):
    def __init__(self,
        input_dim: int,
        output_dim: int,
        action_horizon: int,
        n_layer: int = 7,
        n_head: int = 8,
        n_emb: int = 768,
        max_cond_tokens: int=800,
        p_drop_attn: float = 0.1,
        ) -> None:
        super().__init__()
        
        # input embedding stem
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.randn((1, action_horizon, n_emb)))
        self.time_emb = SinusoidalPosEmb(n_emb)
        # learnable position embedding
        self.cond_pos_emb =  nn.Parameter(torch.randn((1, max_cond_tokens, n_emb)))
        
        # decoder
        # VizTransformerDecoderLayer is a drop-in for nn.TransformerDecoderLayer that
        # additionally captures cross-attention weights (action queries -> cond
        # tokens) into `self.last_attn_weights` when attention_viz.VISUALIZE_ATTENTION
        # is True; zero overhead otherwise (see attention_viz.py).
        decoder_layer = VizTransformerDecoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4*n_emb,
            dropout=p_drop_attn,
            activation='gelu',
            batch_first=True,
            norm_first=True # important for stability
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_layer
        )

        # decoder head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)

        self.action_horizon = action_horizon

        # causal mask over the action-horizon tokens: position i may only attend to
        # positions <= i (matches ImplicitRDP's causal transformer denoiser). Sized to
        # the full action_horizon at construction time and sliced to the actual
        # sequence length in forward() (registered as a non-persistent buffer so it
        # follows .to(device)/.to(dtype) without being saved in checkpoints).
        self.register_buffer(
            "causal_mask",
            nn.Transformer.generate_square_subsequent_mask(action_horizon),
            persistent=False,
        )

        # attention-viz capture: a single snapshot (first decoder layer, last
        # denoising step only), overwritten on every forward() call during a
        # conditional_sample() loop so that whatever is left when the loop ends
        # is the final-step snapshot. Matches ImplicitRDP's Fig. 7 methodology
        # (transformer_for_diffusion.py: `layer == 0 and timestep.item() == 0`).
        # Consumed by pop_attention_viz_capture().
        self._captured_cross_attn_snapshot = None

        # init
        self.apply(self._init_weights)
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def _init_weights(self, module):
        ignore_types = (nn.Dropout, 
            SinusoidalPosEmb, 
            nn.TransformerEncoderLayer, 
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential,
            nn.Embedding)
        if isinstance(module, (nn.Linear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForActionDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_pos_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))
    
    def get_optim_groups(self, weight_decay: float=1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        no_decay.add("_dummy_variable")
        if self.cond_pos_emb is not None:
            no_decay.add("cond_pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def configure_optimizers(self, 
            learning_rate: float=1e-4, 
            weight_decay: float=1e-3,
            betas: Tuple[float, float]=(0.9,0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def forward(self, 
        sample: torch.Tensor, 
        timestep: Union[torch.Tensor, float, int], 
        cond: Optional[torch.Tensor]=None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B,N,n_emb)
        output: (B,T,input_dim)
        """
        
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)
        # (B,1,n_emb)
        
        # 2. process conditions
        cond_emb = torch.cat([cond, time_emb], dim=1)
        tc = cond_emb.shape[1]
        cond_pos_emb = self.cond_pos_emb[
            :, :tc, :
        ]  # each position maps to a (learnable) vector
        cond_emb = cond_emb + cond_pos_emb
        
        # 3. process input
        input_emb = self.input_emb(sample)
        t = input_emb.shape[1]
        pos_emb = self.pos_emb[
            :, :t, :
        ]  # each position maps to a (learnable) vector
        input_emb = input_emb + pos_emb
        
        # 4. transformer
        # causal self-attention over the action-horizon: position i can only see
        # positions <= i. tgt_is_causal=True lets SDPA use the fast causal kernel
        # since causal_mask is exactly the standard upper-triangular -inf mask.
        tgt_mask = self.causal_mask[:t, :t].to(dtype=input_emb.dtype)
        x = self.decoder(
            tgt=input_emb,
            memory=cond_emb,
            tgt_mask=tgt_mask,
            tgt_is_causal=True,
        )
        x = self.ln_f(x)
        x = self.head(x)
        # (B, T, n_out)

        # attention visualization (inference only; no-op when disabled). Only
        # captured outside training. Reads the FIRST decoder layer (matches
        # ImplicitRDP's `layer == 0`) and unconditionally overwrites the
        # snapshot on every forward() call — since conditional_sample() calls
        # forward() once per denoising step in order, whatever remains after
        # the loop ends is exactly the last (final) denoising step's snapshot,
        # matching ImplicitRDP's `timestep.item() == 0` gate without depending
        # on the scheduler's last timestep value actually being 0.
        if not self.training:
            first_layer = self.decoder.layers[0]
            if hasattr(first_layer, "last_attn_weights"):
                # (B, T_action, tc) averaged over heads -> keep batch 0 on CPU
                self._captured_cross_attn_snapshot = (
                    first_layer.last_attn_weights[0].float().cpu()
                )
                del first_layer.last_attn_weights

        return x

    def reset_attention_viz_capture(self):
        """Clear any cross-attention snapshot captured so far. Call before
        starting a fresh conditional_sample() loop so a previous (possibly
        aborted) run can't leak into the next capture."""
        self._captured_cross_attn_snapshot = None

    def pop_attention_viz_capture(self):
        """Return the cross-attention snapshot from the last denoising step of
        the last conditional_sample() call, averaged over the action-horizon
        queries, and clear the internal buffer.

        Snapshot is taken at the FIRST decoder layer and ONLY the LAST
        denoising step (see forward()) — matching ImplicitRDP's Fig. 7
        measurement methodology exactly, so results are directly comparable.

        Returns:
            np.ndarray of shape (n_cond_tokens,) — attention mass each cond
            token (obs tokens..., then the trailing diffusion-timestep token)
            received from action-horizon queries at the final denoising step,
            averaged over the action-horizon queries only. None if nothing was
            captured (VISUALIZE_ATTENTION was off, or this ran in training mode).
        """
        if self._captured_cross_attn_snapshot is None:
            return None
        # (T_action, tc) -> (tc,)
        avg = self._captured_cross_attn_snapshot.mean(dim=0).numpy()
        self._captured_cross_attn_snapshot = None
        return avg


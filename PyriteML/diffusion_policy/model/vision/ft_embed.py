import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)


class FTEmbed(nn.Module):
    def __init__(
        self,
        ft_dim,
        hidden_channels,
        norm_type="group",
        act="gelu",
        alpha_init=1e-2,
        groups=None
    ):
        super().__init__()
        self.ft_dim = ft_dim
        self.hidden_channels = hidden_channels
        self.norm_type = norm_type
        self.groups = groups
        self.alpha_init = alpha_init

        self.conv = nn.Conv1d(
            in_channels=self.ft_dim,
            out_channels=self.hidden_channels,
            kernel_size=3,
            padding=1
        )

        if self.norm_type == "batch":
            self.norm_conv = nn.BatchNorm1d(self.hidden_channels)
        elif self.norm_type == "group":
            if self.groups is None:
                self.groups = self.hidden_channels // 64
            self.norm_conv = nn.GroupNorm(num_groups=self.groups, num_channels=self.hidden_channels)
        else:  # "layer"
            self.norm_conv = nn.LayerNorm(self.hidden_channels)

        if act == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU(inplace=True)

        if self.ft_dim != self.hidden_channels:
            self.res_proj = nn.Linear(self.ft_dim, self.hidden_channels)
        else:
            self.res_proj = nn.Identity()

        self.alpha = nn.Parameter(torch.tensor(self.alpha_init, dtype=torch.float32))

    def forward(self, x):  # x: [B, T, C]
        B, T, C = x.shape

        z = x.transpose(1, 2)   # [B, C, T]
        z = self.conv(z)         # [B, H, T]

        if self.norm_type == "batch" or self.norm_type == "group":
            z = self.norm_conv(z)
            z = self.act(z)
            z = z.transpose(1, 2)   # [B, T, H]
        else:
            z = z.transpose(1, 2)
            z = self.norm_conv(z)
            z = self.act(z)

        return self.res_proj(x) + self.alpha.to(z.dtype) * z  # [B, T, H]


class VectorField(nn.Module):
    def __init__(self, hidden_channels: int, ft_dim: int, act="GELU"):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.ft_dim = ft_dim

        self.time_proj = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.LayerNorm(hidden_channels),
            getattr(nn, act)()
        )

        self.linear = nn.Linear(hidden_channels, hidden_channels * (ft_dim + 1))
        nn.init.xavier_uniform_(self.linear.weight, gain=0.1)
        nn.init.zeros_(self.linear.bias)

        self.norm = nn.LayerNorm(hidden_channels * (ft_dim + 1))
        self.act = getattr(nn, act)()

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)

        if t.dim() == 0:
            t = t.unsqueeze(0).expand(batch_size)
        if t.dim() == 1:
            t_in = t.unsqueeze(-1)
        else:
            t_in = t

        t_emb = self.time_proj(t_in)
        z_time = z + t_emb
        out = self.linear(z_time)
        out = self.norm(out)
        out = self.act(out)
        return out.view(batch_size, self.hidden_channels, self.ft_dim + 1)


class FTNeuralCDEEncoder(nn.Module):
    def __init__(self, ft_dim, embed_dim, initial_dim=None, act="GELU"):
        super().__init__()
        self.ft_dim = ft_dim
        self.embed_dim = embed_dim

        assert initial_dim is not None, "initial_dim must be provided"

        if embed_dim != initial_dim:
            self.initial_encoder = nn.Sequential(
                nn.Linear(initial_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                getattr(nn, act)()
            )
        else:
            self.initial_encoder = None

        self.vector_field = VectorField(embed_dim, ft_dim, act=act)

    def forward(self, img_feat, ft_data, ft_timestamps=None):
        batch_size, seq_len, _ = ft_data.shape

        if ft_timestamps is None:
            ft_timestamps = torch.linspace(0, 1, seq_len, device=ft_data.device)
            ft_timestamps = ft_timestamps.expand(batch_size, seq_len)

        if ft_timestamps.max() > 1.0:
            t_min = ft_timestamps.min(dim=-1, keepdim=True)[0]
            t_max = ft_timestamps.max(dim=-1, keepdim=True)[0]
            ft_timestamps_normalized = (ft_timestamps - t_min) / (t_max - t_min + 1e-10)
        else:
            ft_timestamps_normalized = ft_timestamps

        ft_feat_with_time = torch.cat([
            ft_timestamps_normalized.unsqueeze(-1),
            ft_data
        ], dim=-1)

        import torchcde
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(ft_feat_with_time)
        X = torchcde.CubicSpline(coeffs)

        if self.initial_encoder is not None:
            z0 = self.initial_encoder(img_feat)
        else:
            z0 = img_feat

        z_all = torchcde.cdeint(
            X=X, func=self.vector_field, z0=z0, t=X.interval,
            method='rk4', atol=1e-3, rtol=1e-3,
        )

        if torch.isnan(z_all).any():
            logger.warning("[FT-CDE] z_final contains NaN!")
        if torch.isinf(z_all).any():
            logger.warning("[FT-CDE] z_final contains Inf!")

        return z_all

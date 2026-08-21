from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .analytic_baseline import _task_feedback
from .constants import DEFAULT_SYSTEM


def _transformer(width: int, heads: int, layers: int) -> nn.TransformerEncoder:
    block = nn.TransformerEncoderLayer(
        d_model=width,
        nhead=heads,
        dim_feedforward=4 * width,
        dropout=0.0,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(block, num_layers=layers, enable_nested_tensor=False)


class TaskOrientedEncoder(nn.Module):
    """Compress 12 frequency-group effective channels into 96 complex symbols."""

    def __init__(self, group_size: int = 12, width: int = 128, layers: int = 3, heads: int = 4):
        super().__init__()
        self.group_size = group_size
        self.groups = DEFAULT_SYSTEM.num_downlink_subcarriers // group_size
        self.latent_complex_per_group = DEFAULT_SYSTEM.num_uplink_subcarriers // self.groups
        if self.groups * self.latent_complex_per_group != DEFAULT_SYSTEM.num_uplink_subcarriers:
            raise ValueError("feedback size must be divisible by the number of groups")

        input_features = 2 * DEFAULT_SYSTEM.num_tx_antennas + 2
        self.input = nn.Linear(input_features, width)
        self.position = nn.Parameter(torch.randn(1, self.groups, width) * 0.02)
        self.blocks = _transformer(width, heads, layers)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, 2 * self.latent_complex_per_group)

    def forward(self, channel: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        effective = _task_feedback(channel, self.group_size)
        scale = torch.sqrt(torch.mean(torch.abs(effective).square(), dim=(1, 2), keepdim=True).clamp_min(1e-9))
        normalized = effective / scale
        angle = torch.fft.fft(normalized, dim=-1, norm="ortho")
        log_scale = torch.log(scale).expand(-1, self.groups, 1)
        snr_feature = (snr_db[:, None, None] / 20.0).expand(-1, self.groups, 1)
        features = torch.cat([angle.real, angle.imag, log_scale, snr_feature], dim=-1)
        hidden = self.blocks(self.input(features) + self.position)
        latent = self.output(self.norm(hidden))
        complex_latent = torch.complex(
            latent[..., : self.latent_complex_per_group],
            latent[..., self.latent_complex_per_group :],
        )
        return complex_latent.reshape(channel.shape[0], DEFAULT_SYSTEM.num_uplink_subcarriers)


@dataclass
class FeedbackDecodeResult:
    effective_channel: torch.Tensor
    normalized_angle: torch.Tensor
    log_scale: torch.Tensor


class TaskFeedbackDecoder(nn.Module):
    """Decode noisy DJSCC feedback into task-sufficient effective channels."""

    def __init__(self, group_size: int = 12, width: int = 128, layers: int = 3, heads: int = 4):
        super().__init__()
        self.group_size = group_size
        self.groups = DEFAULT_SYSTEM.num_downlink_subcarriers // group_size
        self.latent_complex_per_group = DEFAULT_SYSTEM.num_uplink_subcarriers // self.groups
        input_features = 2 * self.latent_complex_per_group + 2
        self.input = nn.Linear(input_features, width)
        self.position = nn.Parameter(torch.randn(1, self.groups, width) * 0.02)
        self.blocks = _transformer(width, heads, layers)
        self.norm = nn.LayerNorm(width)
        self.angle_head = nn.Linear(width, 2 * DEFAULT_SYSTEM.num_tx_antennas)
        self.scale_head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, feedback: torch.Tensor, snr_db: torch.Tensor) -> FeedbackDecodeResult:
        batch = feedback.shape[0]
        grouped = feedback.reshape(batch, self.groups, self.latent_complex_per_group)
        snr_feature = (snr_db[:, None, None] / 20.0).expand(-1, self.groups, 1)
        ul_noise_log = ((DEFAULT_SYSTEM.snr_ul_gap_db - snr_db)[:, None, None] / 20.0).expand(-1, self.groups, 1)
        features = torch.cat([grouped.real, grouped.imag, snr_feature, ul_noise_log], dim=-1)
        hidden = self.norm(self.blocks(self.input(features) + self.position))
        angle_values = self.angle_head(hidden)
        angle = torch.complex(
            angle_values[..., : DEFAULT_SYSTEM.num_tx_antennas],
            angle_values[..., DEFAULT_SYSTEM.num_tx_antennas :],
        )
        normalized = torch.fft.ifft(angle, dim=-1, norm="ortho")
        normalized = normalized / torch.sqrt(
            torch.mean(torch.abs(normalized).square(), dim=(1, 2), keepdim=True).clamp_min(1e-9)
        )
        log_scale = self.scale_head(hidden.mean(dim=1)).clamp(-4.0, 4.0)
        effective = normalized * torch.exp(log_scale)[:, None, :]
        return FeedbackDecodeResult(effective, angle, log_scale[:, 0])


def feedback_reconstruction_loss(
    prediction: FeedbackDecodeResult,
    target: torch.Tensor,
    snr_dl_db: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    error = torch.sum(torch.abs(prediction.effective_channel - target).square(), dim=(1, 2))
    energy = torch.sum(torch.abs(target).square(), dim=(1, 2)).clamp_min(1e-9)
    nmse = error / energy

    target_unit = target / torch.linalg.vector_norm(target, dim=-1, keepdim=True).clamp_min(1e-9)
    prediction_unit = prediction.effective_channel / torch.linalg.vector_norm(
        prediction.effective_channel, dim=-1, keepdim=True
    ).clamp_min(1e-9)
    alignment = torch.abs(torch.sum(target_unit.conj() * prediction_unit, dim=-1)).square().clamp(0.0, 1.0)
    beam_loss = 1.0 - alignment.mean(dim=1)

    target_scale = torch.sqrt(torch.mean(torch.abs(target).square(), dim=(1, 2)).clamp_min(1e-9))
    scale_loss = (prediction.log_scale - torch.log(target_scale)).square()

    snr_ul_db = snr_dl_db - DEFAULT_SYSTEM.snr_ul_gap_db
    importance = 0.2 + 0.8 * torch.sigmoid((snr_ul_db + 10.0) / 5.0)
    total_per_sample = nmse + 0.5 * beam_loss + 0.1 * scale_loss
    total = torch.mean(importance * total_per_sample) / torch.mean(importance)
    metrics = {
        "loss": total.detach(),
        "nmse": nmse.mean().detach(),
        "nmse_db": (10.0 * torch.log10(nmse.mean().clamp_min(1e-9))).detach(),
        "beam_alignment": alignment.mean().detach(),
        "scale_mse": scale_loss.mean().detach(),
    }
    return total, metrics


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())

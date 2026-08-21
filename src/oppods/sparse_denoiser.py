from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .constants import DEFAULT_SYSTEM
from .sparse_feedback import DELAY_MODE_ORDER, DELAY_MODE_PRIOR


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


@dataclass
class SparseDenoiseResult:
    effective_channel: torch.Tensor
    coefficients: torch.Tensor
    wiener_coefficients: torch.Tensor


class SparseFeedbackDenoiser(nn.Module):
    """Residual Transformer initialized exactly at the analytic Wiener decoder."""

    def __init__(
        self,
        group_size: int = 12,
        mode_count: int = 6,
        width: int = 128,
        layers: int = 3,
        heads: int = 4,
        wiener_noise_scale: float = 0.75,
    ):
        super().__init__()
        self.group_size = group_size
        self.groups = DEFAULT_SYSTEM.num_downlink_subcarriers // group_size
        self.mode_count = mode_count
        self.wiener_noise_scale = wiener_noise_scale
        prior = torch.tensor(DELAY_MODE_PRIOR[:mode_count], dtype=torch.float32)
        self.register_buffer("mode_prior", prior)

        features = 4 * DEFAULT_SYSTEM.num_tx_antennas + 2
        self.input = nn.Linear(features, width)
        self.position = nn.Parameter(torch.randn(1, mode_count, width) * 0.02)
        self.blocks = _transformer(width, heads, layers)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, 2 * DEFAULT_SYSTEM.num_tx_antennas)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def _wiener(self, selected: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        signal_variance = (
            DEFAULT_SYSTEM.num_uplink_subcarriers
            * self.mode_prior
            / (DEFAULT_SYSTEM.num_tx_antennas * self.mode_prior.sum())
        )
        noise_variance = torch.pow(10.0, -(snr_db - DEFAULT_SYSTEM.snr_ul_gap_db) / 10.0)
        weight = signal_variance[None, :] / (
            signal_variance[None, :] + self.wiener_noise_scale * noise_variance[:, None]
        )
        return selected * weight[:, :, None]

    def forward_details(self, feedback: torch.Tensor, snr_db: torch.Tensor) -> SparseDenoiseResult:
        batch = feedback.shape[0]
        selected = feedback[:, : self.mode_count * DEFAULT_SYSTEM.num_tx_antennas].reshape(
            batch, self.mode_count, DEFAULT_SYSTEM.num_tx_antennas
        )
        wiener = self._wiener(selected, snr_db)
        snr_feature = (snr_db[:, None, None] / 20.0).expand(-1, self.mode_count, 1)
        noise_feature = ((DEFAULT_SYSTEM.snr_ul_gap_db - snr_db)[:, None, None] / 20.0).expand(-1, self.mode_count, 1)
        features = torch.cat(
            [selected.real, selected.imag, wiener.real, wiener.imag, snr_feature, noise_feature], dim=-1
        )
        hidden = self.norm(self.blocks(self.input(features) + self.position))
        residual_values = self.output(hidden)
        residual = torch.complex(
            residual_values[..., : DEFAULT_SYSTEM.num_tx_antennas],
            residual_values[..., DEFAULT_SYSTEM.num_tx_antennas :],
        )
        estimate = wiener + self.residual_scale.tanh() * residual

        coefficients = torch.zeros(
            (batch, self.groups, DEFAULT_SYSTEM.num_tx_antennas),
            device=feedback.device,
            dtype=feedback.dtype,
        )
        coefficients[:, DELAY_MODE_ORDER[: self.mode_count]] = estimate
        angle_group = torch.fft.ifft(coefficients, dim=2, norm="ortho")
        effective = torch.fft.fft(angle_group, dim=1, norm="ortho")
        return SparseDenoiseResult(effective, estimate, wiener)

    def forward(self, feedback: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        return self.forward_details(feedback, snr_db).effective_channel


def sparse_denoising_loss(
    prediction: SparseDenoiseResult,
    clean_coefficients: torch.Tensor,
    target_effective: torch.Tensor,
    snr_dl_db: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    coefficient_error = torch.sum(torch.abs(prediction.coefficients - clean_coefficients).square(), dim=(1, 2))
    coefficient_energy = torch.sum(torch.abs(clean_coefficients).square(), dim=(1, 2)).clamp_min(1e-9)
    coefficient_nmse = coefficient_error / coefficient_energy

    target_unit = target_effective / torch.linalg.vector_norm(target_effective, dim=-1, keepdim=True).clamp_min(1e-9)
    prediction_unit = prediction.effective_channel / torch.linalg.vector_norm(
        prediction.effective_channel, dim=-1, keepdim=True
    ).clamp_min(1e-9)
    alignment = torch.abs(torch.sum(target_unit.conj() * prediction_unit, dim=-1)).square().clamp(0.0, 1.0)
    beam_loss = 1.0 - alignment.mean(dim=1)

    snr_ul_db = snr_dl_db - DEFAULT_SYSTEM.snr_ul_gap_db
    importance = 0.15 + 0.85 * torch.sigmoid((snr_ul_db + 7.0) / 5.0)
    total_per_sample = coefficient_nmse + beam_loss
    total = torch.mean(importance * total_per_sample) / torch.mean(importance)
    metrics = {
        "loss": total.detach(),
        "coefficient_nmse": coefficient_nmse.mean().detach(),
        "coefficient_nmse_db": (10.0 * torch.log10(coefficient_nmse.mean().clamp_min(1e-9))).detach(),
        "beam_alignment": alignment.mean().detach(),
    }
    return total, metrics


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())

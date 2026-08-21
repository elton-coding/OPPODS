from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .channel import add_awgn, apply_downlink, complex_standard_normal, normalize_downlink, normalize_feedback
from .modulation import layered_qam_maxlog_llr, layered_qam_modulate
from .oracle import rzf_precoder
from .pilot_link import _pilot_assignment
from .sparse_denoiser import SparseFeedbackDenoiser
from .sparse_feedback import SparseDelayEncoder


class ReservedPilotMUMIMOLink(nn.Module):
    def __init__(
        self,
        pilot_amplitude: float = math.sqrt(2.0),
        regularization_scale: float = 1.5,
        central_boost: float = -0.5,
    ):
        super().__init__()
        self.pilot_amplitude = pilot_amplitude
        self.regularization_scale = regularization_scale
        self.central_boost = central_boost
        self.encoder = SparseDelayEncoder()
        self.decoder = SparseFeedbackDenoiser()

    def load_denoiser_checkpoint(self, path: str | Path) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.decoder.load_state_dict(checkpoint["denoiser"])
        return checkpoint

    def _receive(
        self,
        received: torch.Tensor,
        snr_db: torch.Tensor,
        own_pilot_index: torch.Tensor,
    ) -> torch.Tensor:
        batch = received.shape[0]
        grouped = received.reshape(batch, 2, 12, 12)
        pilot_vectors = grouped[..., :2].permute(0, 2, 3, 1) / self.pilot_amplitude
        other_pilot_index = 1 - own_pilot_index
        own_gather = own_pilot_index[:, None, None, None].expand(-1, 12, 1, 2)
        other_gather = other_pilot_index[:, None, None, None].expand(-1, 12, 1, 2)
        desired_vector = torch.gather(pilot_vectors, 2, own_gather).squeeze(2)
        other_vector = torch.gather(pilot_vectors, 2, other_gather).squeeze(2)

        matrix = torch.stack([desired_vector, other_vector], dim=-1)
        covariance = matrix @ matrix.conj().transpose(-2, -1)
        noise_variance = torch.pow(10.0, -snr_db / 10.0)
        pilot_estimation_noise = noise_variance / self.pilot_amplitude**2
        identity = torch.eye(2, device=received.device, dtype=received.dtype)
        covariance = covariance + (noise_variance + pilot_estimation_noise)[:, None, None, None] * identity
        weights = torch.linalg.solve(covariance, desired_vector.unsqueeze(-1)).squeeze(-1)
        estimated_vectors = torch.stack([desired_vector, other_vector], dim=2)
        projected = torch.einsum("bgr,bgur->bgu", weights.conj(), estimated_vectors)

        data_y = grouped[..., 2:].reshape(batch, 2, 120)
        weight_sc = weights.repeat_interleave(10, dim=1).permute(0, 2, 1)
        observation = torch.sum(weight_sc.conj() * data_y, dim=1)
        desired_gain = projected[:, :, 0].repeat_interleave(10, dim=1)
        filtered_noise = noise_variance[:, None] * torch.sum(torch.abs(weights).square(), dim=-1)
        residual = torch.abs(projected[:, :, 1]).square() + filtered_noise
        residual = residual.repeat_interleave(10, dim=1)
        return layered_qam_maxlog_llr(
            observation,
            desired_gain,
            residual,
            8,
            central_boost=self.central_boost,
        )

    @torch.no_grad()
    def forward(
        self,
        channel: torch.Tensor,
        bits: torch.Tensor,
        snr_dl: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = channel.shape[0]
        decoded_users = []
        for user in range(2):
            feedback = normalize_feedback(self.encoder(channel[:, user], snr_dl[:, user]))
            noise = complex_standard_normal(tuple(feedback.shape), device=channel.device, generator=generator)
            ul_variance = torch.pow(10.0, -(snr_dl[:, user] - 10.0) / 10.0)
            decoded_users.append(self.decoder(feedback + noise * torch.sqrt(ul_variance)[:, None], snr_dl[:, user]))
        decoded = torch.stack(decoded_users, dim=2)
        precoder_group = rzf_precoder(decoded, torch.pow(10.0, -snr_dl / 10.0), self.regularization_scale)
        precoder_sc = precoder_group.repeat_interleave(12, dim=1)

        streams = torch.zeros((batch, 2, 144), device=channel.device, dtype=channel.dtype)
        data_symbols = torch.stack(
            [layered_qam_modulate(bits[:, user, :960], 8, central_boost=self.central_boost) for user in range(2)],
            dim=1,
        )
        data_positions = (
            12 * torch.arange(12, device=channel.device)[:, None] + torch.arange(2, 12, device=channel.device)
        ).reshape(-1)
        streams[:, :, data_positions] = data_symbols
        actual_pilot_index, inferred_pilot_index, _ = _pilot_assignment(snr_dl)
        pilot0_positions = 12 * torch.arange(12, device=channel.device)
        pilot1_positions = pilot0_positions + 1
        for user in range(2):
            streams[:, user, pilot0_positions] = self.pilot_amplitude * (actual_pilot_index[:, user] == 0)[:, None].to(
                channel.dtype
            )
            streams[:, user, pilot1_positions] = self.pilot_amplitude * (actual_pilot_index[:, user] == 1)[:, None].to(
                channel.dtype
            )

        signal = torch.einsum("bstu,bus->bts", precoder_sc, streams)
        signal, _ = normalize_downlink(signal)
        received, _ = add_awgn(apply_downlink(channel, signal), snr_dl, generator=generator)
        llr = torch.stack(
            [self._receive(received[:, user], snr_dl[:, user], inferred_pilot_index[:, user]) for user in range(2)],
            dim=1,
        )
        identity_accuracy = (inferred_pilot_index == actual_pilot_index).to(torch.float32)
        return llr, identity_accuracy

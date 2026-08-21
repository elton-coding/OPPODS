from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .channel import add_awgn, apply_downlink, complex_standard_normal, normalize_downlink, normalize_feedback
from .modulation import layered_qam_maxlog_llr, layered_qam_modulate
from .oracle import rzf_precoder
from .sparse_denoiser import SparseFeedbackDenoiser
from .sparse_feedback import SparseDelayEncoder


def _pilot_codebook(device: torch.device) -> torch.Tensor:
    sample = torch.arange(12, device=device, dtype=torch.float32)
    return torch.stack(
        [torch.polar(torch.ones_like(sample), 2.0 * math.pi * harmonic * sample / 12.0) for harmonic in (1, 2)]
    )


def _pilot_assignment(snr_db: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign high/low-SNR pilot identities and a 5-bit separability threshold."""
    actual = torch.empty_like(snr_db, dtype=torch.int64)
    user0_is_high = snr_db[:, 0] >= snr_db[:, 1]
    actual[:, 0] = (~user0_is_high).to(torch.int64)
    actual[:, 1] = user0_is_high.to(torch.int64)
    lower_snr = snr_db.min(dim=1).values
    threshold_index = (torch.floor((lower_snr + 20.0) / 1.25).to(torch.int64) + 1).clamp(1, 31)
    threshold = -20.0 + 1.25 * threshold_index.to(torch.float32)
    inferred = (snr_db <= threshold[:, None]).to(torch.int64)
    return actual, inferred, threshold_index


class PilotAidedMUMIMOLink(nn.Module):
    def __init__(
        self,
        pilot_power: float = 0.1,
        regularization_scale: float = 1.5,
        central_boost: float = -0.5,
    ):
        super().__init__()
        if not 0.0 < pilot_power < 1.0:
            raise ValueError("pilot_power must be in (0, 1)")
        self.pilot_power = pilot_power
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
        codebook: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = received.shape[0]
        grouped_y = received.reshape(batch, 2, 12, 12)
        all_estimates = torch.einsum("brgl,cl->bgcr", grouped_y, codebook.conj())
        all_estimates = all_estimates / (12.0 * math.sqrt(self.pilot_power))
        own_gather = own_pilot_index[:, None, None, None].expand(-1, 12, 1, 2)
        other_pilot_index = 1 - own_pilot_index
        other_gather = other_pilot_index[:, None, None, None].expand(-1, 12, 1, 2)
        desired_vector = torch.gather(all_estimates, 2, own_gather).squeeze(2)
        other_vector = torch.gather(all_estimates, 2, other_gather).squeeze(2)
        own_code = codebook[own_pilot_index]
        other_code = codebook[other_pilot_index]

        selected_codes = torch.stack([own_code, other_code], dim=1)
        correlations = torch.einsum("brgl,bul->bgru", grouped_y, selected_codes.conj())
        gram = torch.einsum("bul,bvl->buv", selected_codes, selected_codes.conj())
        joint_estimates = torch.einsum("bgru,buv->bgrv", correlations, torch.linalg.inv(gram))
        joint_estimates = joint_estimates.permute(0, 1, 3, 2) / math.sqrt(self.pilot_power)
        desired_vector = joint_estimates[:, :, 0]
        other_vector = joint_estimates[:, :, 1]

        matrix = torch.stack([desired_vector, other_vector], dim=-1)
        covariance = (1.0 - self.pilot_power) * (matrix @ matrix.conj().transpose(-2, -1))
        noise_variance = torch.pow(10.0, -snr_db / 10.0)
        identity = torch.eye(2, device=received.device, dtype=received.dtype)
        covariance = covariance + noise_variance[:, None, None, None] * identity
        weights = torch.linalg.solve(covariance, desired_vector.unsqueeze(-1)).squeeze(-1)
        estimated_vectors = torch.stack([desired_vector, other_vector], dim=2)
        projected = torch.einsum("bgr,bgur->bgu", weights.conj(), estimated_vectors)
        pilot_group = torch.einsum("bgu,bul->bgl", projected, selected_codes)
        pilot_sc = pilot_group.reshape(batch, 144)
        weight_sc = weights.repeat_interleave(12, dim=1).permute(0, 2, 1)
        observation = torch.sum(weight_sc.conj() * received, dim=1) - math.sqrt(self.pilot_power) * pilot_sc

        desired_projected = projected[:, :, 0]
        other_projected = projected[:, :, 1]
        desired_gain = math.sqrt(1.0 - self.pilot_power) * desired_projected
        filtered_noise = noise_variance[:, None] * torch.sum(torch.abs(weights).square(), dim=-1)
        residual = (1.0 - self.pilot_power) * torch.abs(other_projected).square() + filtered_noise
        desired_gain = desired_gain.repeat_interleave(12, dim=1)
        residual = residual.repeat_interleave(12, dim=1)
        llr = layered_qam_maxlog_llr(
            observation,
            desired_gain,
            residual,
            8,
            central_boost=self.central_boost,
        )
        return llr, other_pilot_index

    @torch.no_grad()
    def forward(
        self,
        channel: torch.Tensor,
        bits: torch.Tensor,
        snr_dl: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        decoded_users = []
        for user in range(2):
            feedback = normalize_feedback(self.encoder(channel[:, user], snr_dl[:, user]))
            noise = complex_standard_normal(tuple(feedback.shape), device=channel.device, generator=generator)
            ul_variance = torch.pow(10.0, -(snr_dl[:, user] - 10.0) / 10.0)
            decoded_users.append(self.decoder(feedback + noise * torch.sqrt(ul_variance)[:, None], snr_dl[:, user]))
        decoded = torch.stack(decoded_users, dim=2)
        precoder_group = rzf_precoder(decoded, torch.pow(10.0, -snr_dl / 10.0), self.regularization_scale)
        precoder_sc = precoder_group.repeat_interleave(12, dim=1)
        symbols = torch.stack(
            [layered_qam_modulate(bits[:, user], 8, central_boost=self.central_boost) for user in range(2)], dim=1
        )
        codebook = _pilot_codebook(channel.device)
        actual_pilot_index, inferred_pilot_index, _ = _pilot_assignment(snr_dl)
        pilots = codebook[actual_pilot_index]
        pilot_sc = pilots.repeat(1, 1, 12)
        streams = math.sqrt(1.0 - self.pilot_power) * symbols + math.sqrt(self.pilot_power) * pilot_sc
        signal = torch.einsum("bstu,bus->bts", precoder_sc, streams)
        signal, _ = normalize_downlink(signal)
        received, _ = add_awgn(apply_downlink(channel, signal), snr_dl, generator=generator)
        results = [
            self._receive(received[:, user], snr_dl[:, user], inferred_pilot_index[:, user], codebook)
            for user in range(2)
        ]
        identity_accuracy = torch.stack(
            [(inferred_pilot_index[:, user] == actual_pilot_index[:, user]).to(torch.float32) for user in range(2)],
            dim=1,
        )
        return torch.stack([result[0] for result in results], dim=1), identity_accuracy

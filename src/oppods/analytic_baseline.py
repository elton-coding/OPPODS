from __future__ import annotations

import math

import torch
from torch import nn

from .channel import add_awgn, apply_downlink, complex_standard_normal, normalize_downlink, normalize_feedback
from .constants import DEFAULT_SYSTEM
from .modulation import layered_qam_maxlog_llr, layered_qam_modulate, qam_constellation
from .oracle import dominant_receive_combiner, grouped_effective_channel


def _task_feedback(channel: torch.Tensor, group_size: int) -> torch.Tensor:
    """Build locally reproducible effective-channel feedback [batch, group, tx]."""
    combiner = dominant_receive_combiner(channel[:, None], group_size)[:, 0]
    _, effective_group = grouped_effective_channel(channel[:, None], combiner[:, None], group_size)
    return effective_group[:, :, 0]


def _feedback_to_mrt(feedback: torch.Tensor) -> torch.Tensor:
    """Convert row-channel feedback to unit-norm MRT columns."""
    return feedback.conj() / torch.linalg.vector_norm(feedback, dim=-1, keepdim=True).clamp_min(1e-9)


def _decision_directed_gain(
    observation: torch.Tensor,
    residual_variance: torch.Tensor,
    *,
    bits_per_symbol: int,
    group_size: int,
    iterations: int,
    central_boost: float,
) -> torch.Tensor:
    """Blindly calibrate the residual complex gain using frequency-group decisions."""
    batch, subcarriers = observation.shape
    groups = subcarriers // group_size
    grouped = observation.reshape(batch, groups, group_size)
    grouped_variance = residual_variance.reshape(batch, groups, group_size)
    signal_power = torch.mean(torch.abs(grouped).square() - grouped_variance, dim=-1, keepdim=True).clamp(0.09, 9.0)
    gain = torch.complex(torch.sqrt(signal_power), torch.zeros_like(signal_power))
    if iterations <= 0:
        return gain.repeat_interleave(group_size, dim=1).reshape(batch, subcarriers)

    points, _ = qam_constellation(bits_per_symbol, device=observation.device, central_boost=central_boost)
    for _ in range(iterations):
        equalized = grouped / gain
        nearest = torch.argmin(torch.abs(equalized[..., None] - points).square(), dim=-1)
        decisions = points[nearest]
        estimate = torch.sum(grouped * decisions.conj(), dim=-1, keepdim=True) / torch.sum(
            torch.abs(decisions).square(), dim=-1, keepdim=True
        ).clamp_min(1e-6)
        magnitude = torch.abs(estimate).clamp(0.3, 3.0)
        phase = torch.angle(estimate).clamp(-math.pi / 3.0, math.pi / 3.0)
        estimate = torch.polar(magnitude, phase)
        gain = 0.5 * gain + 0.5 * estimate
    return gain.repeat_interleave(group_size, dim=1).reshape(batch, subcarriers)


class AnalyticEncoder(nn.Module):
    def __init__(self, group_size: int = 24):
        super().__init__()
        self.group_size = group_size

    def forward(self, channel: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        del snr_db
        feedback = _task_feedback(channel, self.group_size)
        batch = channel.shape[0]
        output = feedback.reshape(batch, -1)
        if output.shape[1] != DEFAULT_SYSTEM.num_uplink_subcarriers:
            raise RuntimeError(f"feedback shape mismatch: {tuple(output.shape)}")
        return output


class AnalyticTransmitter(nn.Module):
    def __init__(
        self,
        group_size: int = 24,
        bits_per_symbol: int = 8,
        central_boost: float = 0.0,
    ):
        super().__init__()
        self.group_size = group_size
        self.bits_per_symbol = bits_per_symbol
        self.central_boost = central_boost
        self.groups = DEFAULT_SYSTEM.num_downlink_subcarriers // group_size

    def forward(
        self,
        bits_list: list[torch.Tensor],
        feedback_list: list[torch.Tensor],
        snr_dl: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del snr_dl
        batch = bits_list[0].shape[0]
        feedback = torch.stack(
            [item.reshape(batch, self.groups, DEFAULT_SYSTEM.num_tx_antennas) for item in feedback_list],
            dim=2,
        )
        beam = _feedback_to_mrt(feedback)
        beam = beam * math.sqrt(1.0 / DEFAULT_SYSTEM.num_ue)
        beam_sc = beam.repeat_interleave(self.group_size, dim=1)

        transmitted_bits = DEFAULT_SYSTEM.num_downlink_subcarriers * self.bits_per_symbol
        symbols = torch.stack(
            [
                layered_qam_modulate(item[:, :transmitted_bits], self.bits_per_symbol, self.central_boost)
                for item in bits_list
            ],
            dim=1,
        )
        signal = torch.einsum("bstu,bus->bts", beam_sc.permute(0, 1, 3, 2), symbols)
        ctrl = torch.zeros(
            (batch, DEFAULT_SYSTEM.num_downlink_ctrl_bits),
            device=signal.device,
            dtype=bits_list[0].dtype,
        )
        return signal, ctrl


class AnalyticReceiver(nn.Module):
    def __init__(
        self,
        group_size: int = 24,
        bits_per_symbol: int = 8,
        decoded_layers: int | None = None,
        interference_scale: float = 1.0,
        use_agc: bool = True,
        decision_directed_iterations: int = 2,
        central_boost: float = 0.0,
    ):
        super().__init__()
        self.group_size = group_size
        self.bits_per_symbol = bits_per_symbol
        self.decoded_layers = decoded_layers or bits_per_symbol
        if not 1 <= self.decoded_layers <= bits_per_symbol:
            raise ValueError("decoded_layers must be in [1, bits_per_symbol]")
        self.interference_scale = interference_scale
        self.use_agc = use_agc
        self.decision_directed_iterations = decision_directed_iterations
        self.central_boost = central_boost
        self.groups = DEFAULT_SYSTEM.num_downlink_subcarriers // group_size

    def forward(
        self,
        received: torch.Tensor,
        channel: torch.Tensor,
        ctrl_bits: torch.Tensor,
        snr_db: torch.Tensor,
    ) -> torch.Tensor:
        del ctrl_bits
        local_feedback = _task_feedback(channel, self.group_size)
        local_feedback = normalize_feedback(local_feedback.reshape(channel.shape[0], -1)).reshape(
            channel.shape[0], self.groups, DEFAULT_SYSTEM.num_tx_antennas
        )
        local_beam = _feedback_to_mrt(local_feedback) * math.sqrt(1.0 / DEFAULT_SYSTEM.num_ue)
        local_beam = local_beam.repeat_interleave(self.group_size, dim=1)

        desired_vector = torch.einsum("brts,bst->brs", channel, local_beam)
        desired_power = torch.sum(torch.abs(desired_vector).square(), dim=1).clamp_min(1e-9)
        spatial_filter = desired_vector / desired_power[:, None, :]
        observation = torch.sum(spatial_filter.conj() * received, dim=1)

        noise_variance = torch.pow(10.0, -snr_db / 10.0)
        filtered_noise = noise_variance[:, None] * torch.sum(torch.abs(spatial_filter).square(), dim=1)
        projected_channel = torch.einsum("brs,brts->bts", spatial_filter.conj(), channel)
        isotropic_interference = (
            0.5 * torch.sum(torch.abs(projected_channel).square(), dim=1) / DEFAULT_SYSTEM.num_tx_antennas
        )
        residual_variance = filtered_noise + self.interference_scale * isotropic_interference

        if self.use_agc:
            gain = _decision_directed_gain(
                observation,
                residual_variance,
                bits_per_symbol=self.bits_per_symbol,
                group_size=self.group_size,
                iterations=self.decision_directed_iterations,
                central_boost=self.central_boost,
            )
        else:
            gain = torch.ones_like(observation)
        llr = layered_qam_maxlog_llr(
            observation,
            gain,
            residual_variance,
            self.bits_per_symbol,
            self.central_boost,
        )
        return llr[:, : DEFAULT_SYSTEM.num_downlink_subcarriers * self.decoded_layers]


class AnalyticMUMIMOLink(nn.Module):
    """Exact local competition wrapper for the deployable analytic baseline."""

    def __init__(
        self,
        group_size: int = 24,
        bits_per_symbol: int = 8,
        decoded_layers: int | None = None,
        interference_scale: float = 1.0,
        use_agc: bool = True,
        decision_directed_iterations: int = 2,
        central_boost: float = 0.0,
    ):
        super().__init__()
        self.encoder = AnalyticEncoder(group_size)
        self.transmitter = AnalyticTransmitter(group_size, bits_per_symbol, central_boost)
        self.receiver = AnalyticReceiver(
            group_size,
            bits_per_symbol,
            decoded_layers,
            interference_scale,
            use_agc,
            decision_directed_iterations,
            central_boost,
        )

    @torch.no_grad()
    def forward(
        self,
        channel: torch.Tensor,
        bits: torch.Tensor,
        snr_dl: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        _, users, _, _, _ = channel.shape
        snr_ul = snr_dl - DEFAULT_SYSTEM.snr_ul_gap_db
        feedback_list: list[torch.Tensor] = []
        for user in range(users):
            feedback = self.encoder(channel[:, user], snr_dl[:, user])
            feedback = normalize_feedback(feedback)
            uplink_noise = (
                complex_standard_normal(tuple(feedback.shape), device=feedback.device, generator=generator)
                * torch.sqrt(torch.pow(10.0, -snr_ul[:, user] / 10.0))[:, None]
            )
            feedback_list.append(feedback + uplink_noise)

        bits_list = [bits[:, user] for user in range(users)]
        signal, ctrl = self.transmitter(bits_list, feedback_list, snr_dl.transpose(0, 1))
        signal, _ = normalize_downlink(signal)
        noiseless = apply_downlink(channel, signal)
        received, _ = add_awgn(noiseless, snr_dl, generator=generator)
        return torch.stack(
            [self.receiver(received[:, user], channel[:, user], ctrl, snr_dl[:, user]) for user in range(users)],
            dim=1,
        )

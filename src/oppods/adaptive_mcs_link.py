from __future__ import annotations

import torch
from torch import nn

from .analytic_baseline import _decision_directed_gain, _feedback_to_mrt, _task_feedback
from .channel import add_awgn, apply_downlink, complex_standard_normal, normalize_downlink, normalize_feedback
from .constants import DEFAULT_SYSTEM
from .denoised_link import DenoisedSparseMUMIMOLink
from .modulation import layered_qam_maxlog_llr, layered_qam_modulate
from .oracle import rzf_precoder

MCS_ORDERS = (1, 2, 4, 6, 8)


def select_mcs(snr_db: torch.Tensor, thresholds: tuple[float, float, float, float]) -> torch.Tensor:
    result = torch.ones_like(snr_db, dtype=torch.int64)
    for threshold, order in zip(thresholds, MCS_ORDERS[1:], strict=True):
        result = torch.where(snr_db >= threshold, order, result)
    return result


class AdaptiveMCSReceiver(nn.Module):
    def __init__(
        self,
        thresholds: tuple[float, float, float, float],
        *,
        group_size: int = 12,
        decision_directed_iterations: int = 15,
        interference_scale: float = 1.0,
        central_boost: float = -0.5,
    ):
        super().__init__()
        self.thresholds = thresholds
        self.group_size = group_size
        self.groups = DEFAULT_SYSTEM.num_downlink_subcarriers // group_size
        self.decision_directed_iterations = decision_directed_iterations
        self.interference_scale = interference_scale
        self.central_boost = central_boost

    def forward(
        self,
        received: torch.Tensor,
        channel: torch.Tensor,
        ctrl_bits: torch.Tensor,
        snr_db: torch.Tensor,
    ) -> torch.Tensor:
        del ctrl_bits
        batch = channel.shape[0]
        local_feedback = _task_feedback(channel, self.group_size)
        local_feedback = normalize_feedback(local_feedback.reshape(batch, -1)).reshape(
            batch, self.groups, DEFAULT_SYSTEM.num_tx_antennas
        )
        local_beam = _feedback_to_mrt(local_feedback) * (1.0 / DEFAULT_SYSTEM.num_ue) ** 0.5
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

        selected_order = select_mcs(snr_db, self.thresholds)
        output = torch.zeros(
            (batch, DEFAULT_SYSTEM.num_downlink_subcarriers * max(MCS_ORDERS)),
            device=received.device,
            dtype=torch.float32,
        )
        for order in MCS_ORDERS:
            boost = self.central_boost if order >= 4 else 0.0
            gain = _decision_directed_gain(
                observation,
                residual_variance,
                bits_per_symbol=order,
                group_size=self.group_size,
                iterations=self.decision_directed_iterations,
                central_boost=boost,
            )
            candidate = layered_qam_maxlog_llr(observation, gain, residual_variance, order, boost)
            length = DEFAULT_SYSTEM.num_downlink_subcarriers * order
            output[:, :length] = torch.where(
                (selected_order == order)[:, None],
                candidate,
                output[:, :length],
            )
        return output


class AdaptiveMCSMUMIMOLink(DenoisedSparseMUMIMOLink):
    def __init__(
        self,
        thresholds: tuple[float, float, float, float] = (-8.0, -3.0, 3.0, 10.0),
        **kwargs: object,
    ):
        if tuple(sorted(thresholds)) != thresholds:
            raise ValueError("MCS thresholds must be ascending")
        super().__init__(**kwargs)
        self.thresholds = thresholds
        self.receiver = AdaptiveMCSReceiver(
            thresholds,
            group_size=self.transmitter.group_size,
            decision_directed_iterations=15,
            central_boost=self.transmitter.central_boost,
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
        feedback_list: list[torch.Tensor] = []
        snr_ul = snr_dl - DEFAULT_SYSTEM.snr_ul_gap_db
        for user in range(DEFAULT_SYSTEM.num_ue):
            feedback = normalize_feedback(self.encoder(channel[:, user], snr_dl[:, user]))
            noise = complex_standard_normal(tuple(feedback.shape), device=feedback.device, generator=generator)
            feedback_list.append(feedback + noise * torch.sqrt(torch.pow(10.0, -snr_ul[:, user] / 10.0))[:, None])

        snr_for_transmitter = snr_dl.transpose(0, 1)
        effective = torch.stack(
            [self.transmitter.decoder(feedback_list[user], snr_for_transmitter[user]) for user in range(2)], dim=2
        )
        noise_variance = torch.pow(10.0, -snr_dl / 10.0)
        beam = rzf_precoder(
            effective,
            noise_variance,
            self.transmitter.regularization_scale,
            self.transmitter.fairness_exponent,
        ).permute(0, 1, 3, 2)
        beam_sc = beam.repeat_interleave(self.transmitter.group_size, dim=1)
        selected_orders = select_mcs(snr_dl, self.thresholds)
        symbols = torch.zeros(
            (channel.shape[0], DEFAULT_SYSTEM.num_ue, DEFAULT_SYSTEM.num_downlink_subcarriers),
            device=channel.device,
            dtype=channel.dtype,
        )
        for order in MCS_ORDERS:
            length = DEFAULT_SYSTEM.num_downlink_subcarriers * order
            boost = self.transmitter.central_boost if order >= 4 else 0.0
            candidate = torch.stack(
                [layered_qam_modulate(bits[:, user, :length], order, boost) for user in range(2)], dim=1
            )
            symbols = torch.where((selected_orders == order)[:, :, None], candidate, symbols)
        signal = torch.einsum("bstu,bus->bts", beam_sc.permute(0, 1, 3, 2), symbols)
        signal, _ = normalize_downlink(signal)
        received, _ = add_awgn(apply_downlink(channel, signal), snr_dl, generator=generator)
        ctrl = torch.zeros(
            (channel.shape[0], DEFAULT_SYSTEM.num_downlink_ctrl_bits),
            device=channel.device,
            dtype=bits.dtype,
        )
        return torch.stack(
            [
                self.receiver(received[:, user], channel[:, user], ctrl, snr_dl[:, user])
                for user in range(DEFAULT_SYSTEM.num_ue)
            ],
            dim=1,
        )

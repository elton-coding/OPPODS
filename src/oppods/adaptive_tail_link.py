from __future__ import annotations

import torch
from torch import nn

from .analytic_baseline import AnalyticReceiver, _feedback_to_mrt, _task_feedback
from .channel import add_awgn, apply_downlink, complex_standard_normal, normalize_downlink, normalize_feedback
from .constants import DEFAULT_SYSTEM
from .denoised_link import DenoisedSparseMUMIMOLink
from .modulation import layered_qam_modulate
from .oracle import rzf_precoder

TAIL_CONTROL_INDEX = 31


def _integer_to_control(value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    shifts = torch.arange(4, -1, -1, device=value.device, dtype=torch.int64)
    return torch.bitwise_and(torch.bitwise_right_shift(value[:, None], shifts[None, :]), 1).to(dtype)


class AdaptiveTailReceiver(nn.Module):
    """Use coherent repetition combining for users in the extreme-SNR tail."""

    def __init__(self, baseline: AnalyticReceiver, tail_threshold_db: float = -10.5, code_length: int = 24):
        super().__init__()
        if DEFAULT_SYSTEM.num_downlink_subcarriers % code_length:
            raise ValueError("code_length must divide the number of downlink subcarriers")
        self.baseline = baseline
        self.tail_threshold_db = tail_threshold_db
        self.code_length = code_length

    def forward(
        self,
        received: torch.Tensor,
        channel: torch.Tensor,
        ctrl_bits: torch.Tensor,
        snr_db: torch.Tensor,
    ) -> torch.Tensor:
        llr = self.baseline(received, channel, ctrl_bits, snr_db)
        control_weights = torch.tensor((16, 8, 4, 2, 1), device=ctrl_bits.device, dtype=torch.int64)
        control_index = torch.sum(ctrl_bits.to(torch.int64) * control_weights[None, :], dim=1)
        tail_mask = (control_index == TAIL_CONTROL_INDEX) & (snr_db < self.tail_threshold_db)
        if not bool(torch.any(tail_mask).item()):
            return llr

        batch = channel.shape[0]
        group_size = self.baseline.group_size
        groups = DEFAULT_SYSTEM.num_downlink_subcarriers // group_size
        local_feedback = _task_feedback(channel, group_size)
        local_feedback = normalize_feedback(local_feedback.reshape(batch, -1)).reshape(
            batch, groups, DEFAULT_SYSTEM.num_tx_antennas
        )
        local_beam = _feedback_to_mrt(local_feedback) * (1.0 / DEFAULT_SYSTEM.num_ue) ** 0.5
        local_beam = local_beam.repeat_interleave(group_size, dim=1)
        desired_vector = torch.einsum("brts,bst->brs", channel, local_beam)
        desired_power = torch.sum(torch.abs(desired_vector).square(), dim=1).clamp_min(1e-9)
        spatial_filter = desired_vector / desired_power[:, None, :]
        observation = torch.sum(spatial_filter.conj() * received, dim=1)
        noise_variance = torch.pow(10.0, -snr_db / 10.0)
        filtered_noise = noise_variance[:, None] * torch.sum(torch.abs(spatial_filter).square(), dim=1)

        projected_channel = torch.einsum("brs,brts->bts", spatial_filter.conj(), channel)
        interference = 0.5 * torch.sum(torch.abs(projected_channel).square(), dim=1) / DEFAULT_SYSTEM.num_tx_antennas
        residual_variance = filtered_noise + interference
        per_re_llr = 2.0 * observation.real / residual_variance.clamp_min(1e-6)
        repetitions = DEFAULT_SYSTEM.num_downlink_subcarriers // self.code_length
        coded_llr = per_re_llr.reshape(batch, repetitions, self.code_length).sum(dim=1)
        llr = llr.clone()
        llr[:, : self.code_length] = torch.where(
            tail_mask[:, None],
            coded_llr,
            llr[:, : self.code_length],
        )
        return llr


class AdaptiveTailMUMIMOLink(DenoisedSparseMUMIMOLink):
    def __init__(self, *, tail_threshold_db: float = -10.5, code_length: int = 24, **kwargs: object):
        super().__init__(**kwargs)
        if DEFAULT_SYSTEM.num_downlink_subcarriers % code_length:
            raise ValueError("code_length must divide the number of downlink subcarriers")
        self.tail_threshold_db = tail_threshold_db
        self.code_length = code_length
        baseline = self.receiver
        self.receiver = AdaptiveTailReceiver(baseline, tail_threshold_db, code_length)

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
        symbols = torch.stack(
            [
                layered_qam_modulate(bits[:, user, :1152], 8, self.transmitter.central_boost)
                for user in range(DEFAULT_SYSTEM.num_ue)
            ],
            dim=1,
        )
        tail_users = snr_dl < self.tail_threshold_db
        repetitions = DEFAULT_SYSTEM.num_downlink_subcarriers // self.code_length
        coded_bits = bits[:, :, : self.code_length].repeat(1, 1, repetitions)
        bpsk_symbols = torch.complex(2.0 * coded_bits - 1.0, torch.zeros_like(symbols.real))
        symbols = torch.where(tail_users[:, :, None], bpsk_symbols, symbols)
        signal = torch.einsum("bstu,bus->bts", beam_sc.permute(0, 1, 3, 2), symbols)
        signal, _ = normalize_downlink(signal)
        profile = torch.where(
            torch.any(tail_users, dim=1),
            torch.full((channel.shape[0],), TAIL_CONTROL_INDEX, device=channel.device, dtype=torch.int64),
            torch.zeros((channel.shape[0],), device=channel.device, dtype=torch.int64),
        )
        ctrl = _integer_to_control(profile, bits.dtype)
        received, _ = add_awgn(apply_downlink(channel, signal), snr_dl, generator=generator)
        return torch.stack(
            [
                self.receiver(received[:, user], channel[:, user], ctrl, snr_dl[:, user])
                for user in range(DEFAULT_SYSTEM.num_ue)
            ],
            dim=1,
        )

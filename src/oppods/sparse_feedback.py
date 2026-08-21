from __future__ import annotations

import math

import torch
from torch import nn

from .analytic_baseline import AnalyticReceiver, _feedback_to_mrt, _task_feedback
from .channel import add_awgn, apply_downlink, complex_standard_normal, normalize_downlink, normalize_feedback
from .constants import DEFAULT_SYSTEM
from .modulation import layered_qam_modulate
from .oracle import rzf_precoder

DELAY_MODE_ORDER = (0, 1, 2, 3, 11, 10)
DELAY_MODE_PRIOR = (0.5722, 0.2068, 0.0698, 0.0301, 0.0424, 0.0173)


class SparseDelayEncoder(nn.Module):
    """Deterministic task-CSI transform using the strongest delay modes."""

    def __init__(self, group_size: int = 12, mode_count: int = 6):
        super().__init__()
        if group_size != 12:
            raise ValueError("the sparse delay code is calibrated for 12 frequency groups")
        if not 1 <= mode_count <= len(DELAY_MODE_ORDER):
            raise ValueError("mode_count must be in [1, 6]")
        self.group_size = group_size
        self.mode_count = mode_count

    def forward(self, channel: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        del snr_db
        effective = _task_feedback(channel, self.group_size)
        coefficients = torch.fft.fft(torch.fft.ifft(effective, dim=1, norm="ortho"), dim=2, norm="ortho")
        selected = coefficients[:, DELAY_MODE_ORDER[: self.mode_count]]
        output = torch.zeros(
            (channel.shape[0], DEFAULT_SYSTEM.num_uplink_subcarriers),
            device=channel.device,
            dtype=channel.dtype,
        )
        output[:, : self.mode_count * DEFAULT_SYSTEM.num_tx_antennas] = selected.flatten(1)
        return output


class SparseDelayDecoder(nn.Module):
    """Wiener-denoise and invert the fixed angle-delay task representation."""

    def __init__(
        self,
        group_size: int = 12,
        mode_count: int = 6,
        use_wiener: bool = True,
        wiener_noise_scale: float = 1.0,
    ):
        super().__init__()
        self.group_size = group_size
        self.groups = DEFAULT_SYSTEM.num_downlink_subcarriers // group_size
        self.mode_count = mode_count
        self.use_wiener = use_wiener
        self.wiener_noise_scale = wiener_noise_scale
        prior = torch.tensor(DELAY_MODE_PRIOR[:mode_count], dtype=torch.float32)
        self.register_buffer("mode_prior", prior)

    def forward(self, feedback: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        batch = feedback.shape[0]
        selected = feedback[:, : self.mode_count * DEFAULT_SYSTEM.num_tx_antennas].reshape(
            batch, self.mode_count, DEFAULT_SYSTEM.num_tx_antennas
        )
        if self.use_wiener:
            # Encoder normalization makes the total energy across the 96 feedback REs equal to 96.
            signal_variance = (
                DEFAULT_SYSTEM.num_uplink_subcarriers
                * self.mode_prior
                / (DEFAULT_SYSTEM.num_tx_antennas * self.mode_prior.sum())
            )
            noise_variance = torch.pow(10.0, -(snr_db - DEFAULT_SYSTEM.snr_ul_gap_db) / 10.0)
            weight = signal_variance[None, :] / (
                signal_variance[None, :] + self.wiener_noise_scale * noise_variance[:, None]
            )
            selected = selected * weight[:, :, None]

        coefficients = torch.zeros(
            (batch, self.groups, DEFAULT_SYSTEM.num_tx_antennas),
            device=feedback.device,
            dtype=feedback.dtype,
        )
        coefficients[:, DELAY_MODE_ORDER[: self.mode_count]] = selected
        angle_group = torch.fft.ifft(coefficients, dim=2, norm="ortho")
        return torch.fft.fft(angle_group, dim=1, norm="ortho")


class SparseDelayTransmitter(nn.Module):
    def __init__(
        self,
        group_size: int = 12,
        mode_count: int = 6,
        bits_per_symbol: int = 8,
        use_wiener: bool = True,
        wiener_noise_scale: float = 1.0,
        precoder: str = "mrt",
        regularization_scale: float = 1.0,
        fairness_exponent: float | None = None,
        central_boost: float = 0.0,
    ):
        super().__init__()
        if precoder not in {"mrt", "rzf"}:
            raise ValueError("precoder must be 'mrt' or 'rzf'")
        self.group_size = group_size
        self.bits_per_symbol = bits_per_symbol
        self.precoder = precoder
        self.regularization_scale = regularization_scale
        self.fairness_exponent = fairness_exponent
        self.central_boost = central_boost
        self.decoder = SparseDelayDecoder(group_size, mode_count, use_wiener, wiener_noise_scale)

    def forward(
        self,
        bits_list: list[torch.Tensor],
        feedback_list: list[torch.Tensor],
        snr_dl: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if snr_dl.shape[0] != DEFAULT_SYSTEM.num_ue:
            snr_dl = snr_dl.transpose(0, 1)
        effective = torch.stack(
            [self.decoder(feedback_list[user], snr_dl[user]) for user in range(DEFAULT_SYSTEM.num_ue)], dim=2
        )
        if self.precoder == "mrt":
            beam = _feedback_to_mrt(effective) * math.sqrt(1.0 / DEFAULT_SYSTEM.num_ue)
        else:
            noise_variance = torch.pow(10.0, -snr_dl.transpose(0, 1) / 10.0)
            beam = rzf_precoder(
                effective,
                noise_variance,
                self.regularization_scale,
                self.fairness_exponent,
            ).permute(0, 1, 3, 2)
        beam_sc = beam.repeat_interleave(self.group_size, dim=1)
        transmitted_bits = DEFAULT_SYSTEM.num_downlink_subcarriers * self.bits_per_symbol
        symbols = torch.stack(
            [
                layered_qam_modulate(bits[:, :transmitted_bits], self.bits_per_symbol, self.central_boost)
                for bits in bits_list
            ],
            dim=1,
        )
        signal = torch.einsum("bstu,bus->bts", beam_sc.permute(0, 1, 3, 2), symbols)
        ctrl = torch.zeros(
            (bits_list[0].shape[0], DEFAULT_SYSTEM.num_downlink_ctrl_bits),
            device=signal.device,
            dtype=bits_list[0].dtype,
        )
        return signal, ctrl


class SparseDelayMUMIMOLink(nn.Module):
    def __init__(
        self,
        group_size: int = 12,
        mode_count: int = 6,
        bits_per_symbol: int = 8,
        use_wiener: bool = True,
        wiener_noise_scale: float = 1.0,
        decision_directed_iterations: int = 6,
        precoder: str = "mrt",
        regularization_scale: float = 1.0,
        fairness_exponent: float | None = None,
        receiver_interference_scale: float = 1.0,
        central_boost: float = 0.0,
    ):
        super().__init__()
        self.encoder = SparseDelayEncoder(group_size, mode_count)
        self.transmitter = SparseDelayTransmitter(
            group_size,
            mode_count,
            bits_per_symbol,
            use_wiener,
            wiener_noise_scale,
            precoder,
            regularization_scale,
            fairness_exponent,
            central_boost,
        )
        self.receiver = AnalyticReceiver(
            group_size,
            bits_per_symbol,
            interference_scale=receiver_interference_scale,
            use_agc=True,
            decision_directed_iterations=decision_directed_iterations,
            central_boost=central_boost,
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
        signal, ctrl = self.transmitter(
            [bits[:, user] for user in range(DEFAULT_SYSTEM.num_ue)], feedback_list, snr_dl.transpose(0, 1)
        )
        signal, _ = normalize_downlink(signal)
        received, _ = add_awgn(apply_downlink(channel, signal), snr_dl, generator=generator)
        return torch.stack(
            [
                self.receiver(received[:, user], channel[:, user], ctrl, snr_dl[:, user])
                for user in range(DEFAULT_SYSTEM.num_ue)
            ],
            dim=1,
        )

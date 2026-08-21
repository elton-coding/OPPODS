from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .analytic_baseline import AnalyticReceiver, _feedback_to_mrt
from .channel import add_awgn, apply_downlink, complex_standard_normal, normalize_downlink, normalize_feedback
from .constants import DEFAULT_SYSTEM
from .feedback_model import TaskFeedbackDecoder, TaskOrientedEncoder
from .modulation import layered_qam_modulate


class LearnedFeedbackTransmitter(nn.Module):
    """Decode task feedback and apply frequency-grouped MRT."""

    def __init__(
        self,
        group_size: int = 12,
        bits_per_symbol: int = 8,
        width: int = 128,
        layers: int = 3,
        heads: int = 4,
    ):
        super().__init__()
        self.group_size = group_size
        self.bits_per_symbol = bits_per_symbol
        self.groups = DEFAULT_SYSTEM.num_downlink_subcarriers // group_size
        self.decoder = TaskFeedbackDecoder(group_size, width, layers, heads)

    def forward(
        self,
        bits_list: list[torch.Tensor],
        feedback_list: list[torch.Tensor],
        snr_dl: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The official interface supplies SNR as [user,batch].
        if snr_dl.shape[0] != DEFAULT_SYSTEM.num_ue:
            snr_dl = snr_dl.transpose(0, 1)
        decoded = [
            self.decoder(feedback_list[user], snr_dl[user]).effective_channel for user in range(DEFAULT_SYSTEM.num_ue)
        ]
        effective = torch.stack(decoded, dim=2)
        beam = _feedback_to_mrt(effective) * math.sqrt(1.0 / DEFAULT_SYSTEM.num_ue)
        beam_sc = beam.repeat_interleave(self.group_size, dim=1)

        transmitted_bits = DEFAULT_SYSTEM.num_downlink_subcarriers * self.bits_per_symbol
        symbols = torch.stack(
            [layered_qam_modulate(bits[:, :transmitted_bits], self.bits_per_symbol) for bits in bits_list], dim=1
        )
        signal = torch.einsum("bstu,bus->bts", beam_sc.permute(0, 1, 3, 2), symbols)
        ctrl = torch.zeros(
            (bits_list[0].shape[0], DEFAULT_SYSTEM.num_downlink_ctrl_bits),
            device=signal.device,
            dtype=bits_list[0].dtype,
        )
        return signal, ctrl


class LearnedFeedbackMUMIMOLink(nn.Module):
    """Local competition wrapper for pretrained DJSCC feedback plus analytic data link."""

    def __init__(
        self,
        group_size: int = 12,
        bits_per_symbol: int = 8,
        width: int = 128,
        layers: int = 3,
        heads: int = 4,
        decoded_layers: int | None = None,
        interference_scale: float = 1.0,
        decision_directed_iterations: int = 6,
    ):
        super().__init__()
        self.encoder = TaskOrientedEncoder(group_size, width, layers, heads)
        self.transmitter = LearnedFeedbackTransmitter(group_size, bits_per_symbol, width, layers, heads)
        self.receiver = AnalyticReceiver(
            group_size,
            bits_per_symbol,
            decoded_layers,
            interference_scale,
            use_agc=True,
            decision_directed_iterations=decision_directed_iterations,
        )

    def load_feedback_checkpoint(self, path: str | Path) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.transmitter.decoder.load_state_dict(checkpoint["decoder"])
        return checkpoint

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
            noise = noise * torch.sqrt(torch.pow(10.0, -snr_ul[:, user] / 10.0))[:, None]
            feedback_list.append(feedback + noise)

        signal, ctrl = self.transmitter(
            [bits[:, user] for user in range(DEFAULT_SYSTEM.num_ue)],
            feedback_list,
            snr_dl.transpose(0, 1),
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

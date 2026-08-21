from __future__ import annotations

import torch

from .analytic_baseline import AnalyticReceiver, _decision_directed_gain, _feedback_to_mrt, _task_feedback
from .channel import normalize_feedback
from .constants import DEFAULT_SYSTEM
from .modulation import layered_qam_maxlog_llr


class MVDRAnalyticReceiver(AnalyticReceiver):
    """Blind two-antenna receiver using a local sample-covariance MVDR filter.

    The receiver still reconstructs its steering vector exclusively from its own
    channel, as required by the competition interface.  The received downlink
    samples provide a group-local covariance estimate that can suppress the
    other user's spatial component without revealing transmitter-side CSI.
    """

    def __init__(
        self,
        *args: object,
        trace_loading: float = 0.1,
        noise_loading: float = 1.0,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        if trace_loading < 0.0 or noise_loading < 0.0:
            raise ValueError("covariance loadings must be non-negative")
        self.trace_loading = trace_loading
        self.noise_loading = noise_loading

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

        grouped_received = received.reshape(batch, DEFAULT_SYSTEM.num_rx_antennas, self.groups, self.group_size)
        grouped_received = grouped_received.permute(0, 2, 1, 3)
        covariance = torch.einsum(
            "bgrl,bgql->bgrq", grouped_received, grouped_received.conj()
        ) / self.group_size
        covariance_trace = covariance.diagonal(dim1=-2, dim2=-1).real.sum(dim=-1)
        noise_variance = torch.pow(10.0, -snr_db / 10.0)
        diagonal_load = (
            self.trace_loading * covariance_trace / DEFAULT_SYSTEM.num_rx_antennas
            + self.noise_loading * noise_variance[:, None]
        )
        identity = torch.eye(
            DEFAULT_SYSTEM.num_rx_antennas,
            dtype=covariance.dtype,
            device=covariance.device,
        )
        loaded_covariance = covariance + diagonal_load[..., None, None] * identity

        grouped_desired = desired_vector.reshape(
            batch, DEFAULT_SYSTEM.num_rx_antennas, self.groups, self.group_size
        ).permute(0, 2, 1, 3)
        grouped_filter = torch.linalg.solve(loaded_covariance, grouped_desired)
        constraint_gain = torch.sum(grouped_filter.conj() * grouped_desired, dim=2, keepdim=True)
        safe_gain = torch.where(
            torch.abs(constraint_gain) > 1e-9,
            constraint_gain.conj(),
            torch.ones_like(constraint_gain),
        )
        grouped_filter = grouped_filter / safe_gain
        spatial_filter = grouped_filter.permute(0, 2, 1, 3).reshape(
            batch, DEFAULT_SYSTEM.num_rx_antennas, DEFAULT_SYSTEM.num_downlink_subcarriers
        )
        observation = torch.sum(spatial_filter.conj() * received, dim=1)

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

from __future__ import annotations

import math
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .analytic_baseline import _feedback_to_mrt, _task_feedback
from .channel import add_awgn, apply_downlink, complex_standard_normal, normalize_downlink, normalize_feedback
from .modulation import layered_qam_maxlog_llr, layered_qam_modulate, qam_constellation
from .oracle import rzf_precoder
from .pilot_link import _pilot_assignment
from .sparse_denoiser import SparseFeedbackDenoiser
from .sparse_feedback import SparseDelayEncoder

PAIR_SEPARATORS = ("pair", "tri3", "binom5", "binom7", "spectral4", "spectral6")
IDENTITY_MODES = (
    "threshold",
    "local",
    "hybrid",
    "safe_hybrid",
    "window_hybrid",
    "asymmetric_hybrid",
)
IDENTITY_SCORE_MODES = ("mean", "local_power", "pilot_power", "joint_power", "global", "median")
FREQUENCY_INTERPOLATIONS = (
    "none",
    "linear",
    "gain_linear",
    "gain_linear_post",
    "gain_phase_post",
    "gain_magnitude_post",
    "gain_polar_post",
)


def _pilot_phase_constellation(
    bits: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels_int = torch.arange(2**bits, device=device, dtype=torch.int64)
    binary = labels_int.clone()
    shifted = labels_int.clone()
    while torch.any(shifted > 0):
        shifted = torch.bitwise_right_shift(shifted, 1)
        binary = torch.bitwise_xor(binary, shifted)
    angle = 2.0 * math.pi * binary.to(torch.float32) / (2**bits)
    points = torch.polar(torch.ones_like(angle), angle).to(dtype=dtype)
    shifts = torch.arange(bits - 1, -1, -1, device=device, dtype=torch.int64)
    labels = torch.bitwise_and(
        torch.bitwise_right_shift(labels_int[:, None], shifts[None]), 1
    ).to(torch.bool)
    return points, labels


def _pilot_slope_constellation(
    bits: int,
    step: float,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels_int = torch.arange(2**bits, device=device, dtype=torch.int64)
    binary = labels_int.clone()
    shifted = labels_int.clone()
    while torch.any(shifted > 0):
        shifted = torch.bitwise_right_shift(shifted, 1)
        binary = torch.bitwise_xor(binary, shifted)
    slopes = (2.0 * binary.to(torch.float32) - (2**bits - 1)) * step
    shifts = torch.arange(bits - 1, -1, -1, device=device, dtype=torch.int64)
    labels = torch.bitwise_and(
        torch.bitwise_right_shift(labels_int[:, None], shifts[None]), 1
    ).to(torch.bool)
    return slopes, labels


class PairedPilotMUMIMOLink(nn.Module):
    """One Walsh-coded pilot RE per group, separated over adjacent groups."""

    def __init__(
        self,
        pilot_amplitude: float = math.sqrt(3.0),
        regularization_scale: float = 1.5,
        central_boost: float = -0.5,
        separator: str = "pair",
        steering_shrinkage: float = 0.0,
        covariance_loading_scale: float = 1.0,
        gain_refinement_iterations: int = 0,
        gain_refinement_min_snr: float = 0.0,
        gain_refinement_rate: float = 0.5,
        gain_refinement_soft_temperature: float = 0.0,
        sample_covariance_blend: float = 0.0,
        fairness_exponent: float | None = None,
        pilot_offset: int = 0,
        identity_mode: str = "threshold",
        identity_margin: float = 0.0,
        identity_margin_snr_slope: float = 0.0,
        identity_margin_bin_slope: float = 0.0,
        identity_score_mode: str = "mean",
        identity_window: float = 1.25,
        identity_margin_to_zero: float = 0.3,
        identity_margin_to_one: float = 0.3,
        control_levels: int = 31,
        control_companding: float = 1.0,
        tri3_side_weight: float = 0.25,
        frequency_interpolation: str = "none",
        gain_interpolation_scale: float = 1.0,
        gain_interpolation_snr_slope: float = 0.0,
        data_gain_refinement_scale: float = 0.0,
        data_gain_refinement_radius: int = 4,
        data_gain_refinement_min_snr: float = -10.0,
        data_gain_kernel: str = "uniform",
        data_gain_center_weight: float = 1.0,
        data_gain_model: str = "constant",
        data_gain_soft_temperature: float = 0.0,
        data_gain_soft_temperature_decay: float = 1.0,
        data_gain_residual_mode: str = "initial",
        data_gain_refinement_iterations: int = 1,
        data_gain_refinement_snr_slope: float = 0.0,
        data_vector_refinement_scale: float = 0.0,
        data_vector_refinement_min_snr: float = -5.0,
        data_vector_smoothing_side: float = 0.0,
        data_vector_refinement_snr_slope: float = 0.0,
        pre_vector_gain_refinement_iterations: int = 0,
        data_vector_soft_temperature: float = 0.0,
        data_vector_confidence_floor: float = 1.0,
        data_vector_reliability_power: float = 0.0,
        data_vector_refinement_iterations: int = 1,
        interference_cancellation_scale: float = 0.0,
        interference_cancellation_temperature: float = 1.0,
        interference_cancellation_min_snr: float = 0.0,
        interference_cancellation_confidence_floor: float = 1.0,
        interference_gain_refinement_scale: float = 0.0,
        interference_gain_refinement_iterations: int = 1,
        interference_cancellation_snr_slope: float = 0.0,
        reciprocal_cancellation_scale: float = 0.0,
        reciprocal_cancellation_temperature: float = 1.0,
        interference_filter_loading_scale: float = 1.0,
        interference_vector_refinement_scale: float = 0.0,
        joint_detection_candidates: int = 0,
        joint_detection_prior_scale: float = 1.0,
        joint_detection_min_snr: float = 0.0,
        pilot_bit_min_snr: float = 100.0,
        pilot_phase_bits: int = 1,
        pilot_phase_schedule: tuple[tuple[float, int], ...] | None = None,
        pilot_phase_weight_power: float = 1.0,
        pilot_phase_segments: int = 1,
        pilot_slope_bits: int = 0,
        pilot_slope_min_snr: float = 100.0,
        pilot_slope_step: float = math.pi / 24.0,
        pilot_phase_gate_threshold: float = 0.0,
    ):
        super().__init__()
        if separator not in PAIR_SEPARATORS:
            raise ValueError(f"separator must be one of {PAIR_SEPARATORS}")
        if not 0.0 <= steering_shrinkage <= 1.0:
            raise ValueError("steering_shrinkage must be in [0, 1]")
        if covariance_loading_scale < 0.0:
            raise ValueError("covariance_loading_scale must be non-negative")
        if gain_refinement_iterations < 0:
            raise ValueError("gain_refinement_iterations must be non-negative")
        if not 0.0 < gain_refinement_rate <= 1.0:
            raise ValueError("gain_refinement_rate must be in (0, 1]")
        if gain_refinement_soft_temperature < 0.0:
            raise ValueError("gain_refinement_soft_temperature must be non-negative")
        if not 0.0 <= sample_covariance_blend <= 1.0:
            raise ValueError("sample_covariance_blend must be in [0, 1]")
        if not 0 <= pilot_offset < 12:
            raise ValueError("pilot_offset must be in [0, 11]")
        if identity_mode not in IDENTITY_MODES:
            raise ValueError(f"identity_mode must be one of {IDENTITY_MODES}")
        if identity_margin < 0.0:
            raise ValueError("identity_margin must be non-negative")
        if identity_score_mode not in IDENTITY_SCORE_MODES:
            raise ValueError(f"identity_score_mode must be one of {IDENTITY_SCORE_MODES}")
        if identity_window < 0.0:
            raise ValueError("identity_window must be non-negative")
        if identity_margin_to_zero < 0.0 or identity_margin_to_one < 0.0:
            raise ValueError("asymmetric identity margins must be non-negative")
        if not 2 <= control_levels <= 32:
            raise ValueError("control_levels must be in [2, 32]")
        if control_companding <= 0.0:
            raise ValueError("control_companding must be positive")
        if not 0.0 <= tri3_side_weight <= 0.5:
            raise ValueError("tri3_side_weight must be in [0, 0.5]")
        if frequency_interpolation not in FREQUENCY_INTERPOLATIONS:
            raise ValueError(f"frequency_interpolation must be one of {FREQUENCY_INTERPOLATIONS}")
        if gain_interpolation_scale < 0.0:
            raise ValueError("gain_interpolation_scale must be non-negative")
        if data_gain_refinement_scale < 0.0:
            raise ValueError("data_gain_refinement_scale must be non-negative")
        if data_gain_refinement_radius < 1:
            raise ValueError("data_gain_refinement_radius must be positive")
        if data_gain_kernel not in {"uniform", "triangular", "binom5"}:
            raise ValueError("data_gain_kernel must be uniform, triangular, or binom5")
        if data_gain_center_weight < 0.0:
            raise ValueError("data_gain_center_weight must be non-negative")
        if data_gain_model not in {"constant", "linear"}:
            raise ValueError("data_gain_model must be constant or linear")
        if data_gain_soft_temperature < 0.0:
            raise ValueError("data_gain_soft_temperature must be non-negative")
        if data_gain_soft_temperature_decay <= 0.0:
            raise ValueError("data_gain_soft_temperature_decay must be positive")
        if data_gain_residual_mode not in {"initial", "current"}:
            raise ValueError("data_gain_residual_mode must be initial or current")
        if data_gain_refinement_iterations < 1:
            raise ValueError("data_gain_refinement_iterations must be positive")
        if data_vector_refinement_scale < 0.0:
            raise ValueError("data_vector_refinement_scale must be non-negative")
        if not 0.0 <= data_vector_smoothing_side <= 0.5:
            raise ValueError("data_vector_smoothing_side must be in [0, 0.5]")
        if pre_vector_gain_refinement_iterations < 0:
            raise ValueError("pre_vector_gain_refinement_iterations must be non-negative")
        if data_vector_soft_temperature < 0.0:
            raise ValueError("data_vector_soft_temperature must be non-negative")
        if not 0.0 <= data_vector_confidence_floor <= 1.0:
            raise ValueError("data_vector_confidence_floor must be in [0, 1]")
        if data_vector_reliability_power < 0.0:
            raise ValueError("data_vector_reliability_power must be non-negative")
        if data_vector_refinement_iterations < 1:
            raise ValueError("data_vector_refinement_iterations must be positive")
        if not 0.0 <= interference_cancellation_scale <= 1.0:
            raise ValueError("interference_cancellation_scale must be in [0, 1]")
        if interference_cancellation_temperature <= 0.0:
            raise ValueError("interference_cancellation_temperature must be positive")
        if not 0.0 <= interference_cancellation_confidence_floor <= 1.0:
            raise ValueError("interference_cancellation_confidence_floor must be in [0, 1]")
        if not 0.0 <= interference_gain_refinement_scale <= 1.0:
            raise ValueError("interference_gain_refinement_scale must be in [0, 1]")
        if interference_gain_refinement_iterations < 1:
            raise ValueError("interference_gain_refinement_iterations must be positive")
        if not 0.0 <= reciprocal_cancellation_scale <= 1.0:
            raise ValueError("reciprocal_cancellation_scale must be in [0, 1]")
        if reciprocal_cancellation_temperature <= 0.0:
            raise ValueError("reciprocal_cancellation_temperature must be positive")
        if interference_filter_loading_scale < 0.0:
            raise ValueError("interference_filter_loading_scale must be non-negative")
        if not 0.0 <= interference_vector_refinement_scale <= 1.0:
            raise ValueError("interference_vector_refinement_scale must be in [0, 1]")
        if joint_detection_candidates not in {0, 1, 2, 4, 8}:
            raise ValueError("joint_detection_candidates must be 0, 1, 2, 4, or 8")
        if joint_detection_prior_scale < 0.0:
            raise ValueError("joint_detection_prior_scale must be non-negative")
        if pilot_phase_schedule is None:
            pilot_phase_schedule = ((pilot_bit_min_snr, pilot_phase_bits),)
        if not pilot_phase_schedule:
            raise ValueError("pilot_phase_schedule must contain at least one level")
        phase_thresholds = tuple(level[0] for level in pilot_phase_schedule)
        phase_bit_counts = tuple(level[1] for level in pilot_phase_schedule)
        if any(bits not in {1, 2, 3, 4, 5, 6} for bits in phase_bit_counts):
            raise ValueError("pilot phase bit counts must be between 1 and 6")
        if any(right <= left for left, right in pairwise(phase_thresholds)):
            raise ValueError("pilot phase SNR thresholds must be strictly increasing")
        if any(right <= left for left, right in pairwise(phase_bit_counts)):
            raise ValueError("pilot phase bit counts must be strictly increasing")
        if pilot_phase_weight_power < 0.0:
            raise ValueError("pilot_phase_weight_power must be non-negative")
        if pilot_phase_segments not in {1, 2, 3, 4, 6, 12}:
            raise ValueError("pilot_phase_segments must divide the 12 pilot groups")
        if pilot_slope_bits not in {0, 1, 2, 3}:
            raise ValueError("pilot_slope_bits must be between 0 and 3")
        if pilot_slope_bits and pilot_phase_segments != 1:
            raise ValueError("pilot slope coding currently requires one phase segment")
        if pilot_slope_bits and pilot_slope_min_snr < phase_thresholds[-1]:
            raise ValueError("pilot slope coding must start after the maximum phase order")
        if pilot_slope_step <= 0.0:
            raise ValueError("pilot_slope_step must be positive")
        if not 0.0 <= pilot_phase_gate_threshold <= 1.0:
            raise ValueError("pilot_phase_gate_threshold must be in [0, 1]")
        self.pilot_amplitude = pilot_amplitude
        self.regularization_scale = regularization_scale
        self.central_boost = central_boost
        self.separator = separator
        self.steering_shrinkage = steering_shrinkage
        self.covariance_loading_scale = covariance_loading_scale
        self.gain_refinement_iterations = gain_refinement_iterations
        self.gain_refinement_min_snr = gain_refinement_min_snr
        self.gain_refinement_rate = gain_refinement_rate
        self.gain_refinement_soft_temperature = gain_refinement_soft_temperature
        self.sample_covariance_blend = sample_covariance_blend
        self.fairness_exponent = fairness_exponent
        self.pilot_offset = pilot_offset
        self.identity_mode = identity_mode
        self.identity_margin = identity_margin
        self.identity_margin_snr_slope = identity_margin_snr_slope
        self.identity_margin_bin_slope = identity_margin_bin_slope
        self.identity_score_mode = identity_score_mode
        self.identity_window = identity_window
        self.identity_margin_to_zero = identity_margin_to_zero
        self.identity_margin_to_one = identity_margin_to_one
        self.control_levels = control_levels
        self.control_companding = control_companding
        self.tri3_side_weight = tri3_side_weight
        self.frequency_interpolation = frequency_interpolation
        self.gain_interpolation_scale = gain_interpolation_scale
        self.gain_interpolation_snr_slope = gain_interpolation_snr_slope
        self.data_gain_refinement_scale = data_gain_refinement_scale
        self.data_gain_refinement_radius = data_gain_refinement_radius
        self.data_gain_refinement_min_snr = data_gain_refinement_min_snr
        self.data_gain_kernel = data_gain_kernel
        self.data_gain_center_weight = data_gain_center_weight
        self.data_gain_model = data_gain_model
        self.data_gain_soft_temperature = data_gain_soft_temperature
        self.data_gain_soft_temperature_decay = data_gain_soft_temperature_decay
        self.data_gain_residual_mode = data_gain_residual_mode
        self.data_gain_refinement_iterations = data_gain_refinement_iterations
        self.data_gain_refinement_snr_slope = data_gain_refinement_snr_slope
        self.data_vector_refinement_scale = data_vector_refinement_scale
        self.data_vector_refinement_min_snr = data_vector_refinement_min_snr
        self.data_vector_smoothing_side = data_vector_smoothing_side
        self.data_vector_refinement_snr_slope = data_vector_refinement_snr_slope
        self.pre_vector_gain_refinement_iterations = pre_vector_gain_refinement_iterations
        self.data_vector_soft_temperature = data_vector_soft_temperature
        self.data_vector_confidence_floor = data_vector_confidence_floor
        self.data_vector_reliability_power = data_vector_reliability_power
        self.data_vector_refinement_iterations = data_vector_refinement_iterations
        self.interference_cancellation_scale = interference_cancellation_scale
        self.interference_cancellation_temperature = interference_cancellation_temperature
        self.interference_cancellation_min_snr = interference_cancellation_min_snr
        self.interference_cancellation_confidence_floor = interference_cancellation_confidence_floor
        self.interference_gain_refinement_scale = interference_gain_refinement_scale
        self.interference_gain_refinement_iterations = interference_gain_refinement_iterations
        self.interference_cancellation_snr_slope = interference_cancellation_snr_slope
        self.reciprocal_cancellation_scale = reciprocal_cancellation_scale
        self.reciprocal_cancellation_temperature = reciprocal_cancellation_temperature
        self.interference_filter_loading_scale = interference_filter_loading_scale
        self.interference_vector_refinement_scale = interference_vector_refinement_scale
        self.joint_detection_candidates = joint_detection_candidates
        self.joint_detection_prior_scale = joint_detection_prior_scale
        self.joint_detection_min_snr = joint_detection_min_snr
        self.pilot_phase_schedule = tuple(pilot_phase_schedule)
        self.pilot_bit_min_snr = self.pilot_phase_schedule[0][0]
        self.pilot_phase_bits = self.pilot_phase_schedule[-1][1]
        self.pilot_phase_weight_power = pilot_phase_weight_power
        self.pilot_phase_segments = pilot_phase_segments
        self.pilot_slope_bits = pilot_slope_bits
        self.pilot_slope_min_snr = pilot_slope_min_snr
        self.pilot_slope_step = pilot_slope_step
        self.pilot_phase_gate_threshold = pilot_phase_gate_threshold
        self.encoder = SparseDelayEncoder()
        self.decoder = SparseFeedbackDenoiser()

    def load_denoiser_checkpoint(self, path: str | Path) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.decoder.load_state_dict(checkpoint["denoiser"])
        return checkpoint

    def _receive(
        self,
        received: torch.Tensor,
        channel: torch.Tensor,
        snr_db: torch.Tensor,
        own_code_index: torch.Tensor,
        identity_threshold: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = received.shape[0]
        grouped = received.reshape(batch, 2, 12, 12)
        pilot = grouped[..., self.pilot_offset].permute(0, 2, 1) / self.pilot_amplitude
        walsh_sign = torch.where(
            torch.arange(12, device=received.device) % 2 == 0,
            torch.ones(12, device=received.device),
            -torch.ones(12, device=received.device),
        ).to(received.dtype)
        if self.separator == "pair":
            paired = pilot.reshape(batch, 6, 2, 2)
            code0_pair = 0.5 * (paired[:, :, 0] + paired[:, :, 1])
            code1_pair = 0.5 * (paired[:, :, 0] - paired[:, :, 1])
            code0 = code0_pair.repeat_interleave(2, dim=1)
            code1 = code1_pair.repeat_interleave(2, dim=1)
            filter_energy = 0.5
        elif self.separator in {"tri3", "binom5", "binom7"}:
            if self.separator == "tri3":
                coefficients = (
                    self.tri3_side_weight,
                    1.0 - 2.0 * self.tri3_side_weight,
                    self.tri3_side_weight,
                )
            elif self.separator == "binom5":
                coefficients = (1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16)
            else:
                coefficients = (1 / 64, 6 / 64, 15 / 64, 20 / 64, 15 / 64, 6 / 64, 1 / 64)

            def smooth(values: torch.Tensor) -> torch.Tensor:
                center = len(coefficients) // 2
                return sum(
                    coefficient * torch.roll(values, center - index, dims=1)
                    for index, coefficient in enumerate(coefficients)
                )

            code0 = smooth(pilot)
            code1 = smooth(pilot * walsh_sign[None, :, None])
            filter_energy = sum(coefficient**2 for coefficient in coefficients)
        else:
            mode_count = int(self.separator.removeprefix("spectral"))
            mode_order = (0, 1, 11, 2, 10, 3)
            mask = torch.zeros(12, device=received.device, dtype=received.real.dtype)
            mask[list(mode_order[:mode_count])] = 1.0

            def spectral(values: torch.Tensor) -> torch.Tensor:
                coefficients = torch.fft.ifft(values, dim=1, norm="ortho")
                return torch.fft.fft(coefficients * mask[None, :, None], dim=1, norm="ortho")

            code0 = spectral(pilot)
            code1 = spectral(pilot * walsh_sign[None, :, None])
            filter_energy = mode_count / 12.0
        vectors = torch.stack([code0, code1], dim=2)
        local_vector = None
        if self.identity_mode != "threshold" or self.steering_shrinkage > 0.0:
            local_feedback = _task_feedback(channel, 12)
            local_feedback = normalize_feedback(local_feedback.reshape(batch, -1)).reshape(batch, 12, 16)
            local_beam = _feedback_to_mrt(local_feedback) * math.sqrt(0.5)
            pilot_positions = 12 * torch.arange(12, device=channel.device) + self.pilot_offset
            pilot_channel = channel[..., pilot_positions]
            local_vector = torch.einsum("brtg,bgt->bgr", pilot_channel, local_beam)
        selected_code_index = own_code_index
        if self.identity_mode != "threshold":
            assert local_vector is not None
            inner = torch.sum(vectors.conj() * local_vector[:, :, None, :], dim=-1)
            denominator = torch.sum(torch.abs(vectors).square(), dim=-1) * torch.sum(
                torch.abs(local_vector).square(), dim=-1
            )[:, :, None]
            group_similarity = torch.abs(inner).square() / denominator.clamp_min(1e-9)
            if self.identity_score_mode == "mean":
                similarity = group_similarity.mean(dim=1)
            elif self.identity_score_mode == "median":
                similarity = group_similarity.median(dim=1).values
            elif self.identity_score_mode == "global":
                similarity = torch.abs(inner).square().sum(dim=1) / denominator.sum(dim=1).clamp_min(1e-9)
            else:
                local_power = torch.sum(torch.abs(local_vector).square(), dim=-1)[:, :, None]
                pilot_power = torch.sum(torch.abs(vectors).square(), dim=-1)
                if self.identity_score_mode == "local_power":
                    score_weight = local_power.expand_as(group_similarity)
                elif self.identity_score_mode == "pilot_power":
                    score_weight = pilot_power
                else:
                    score_weight = torch.sqrt((local_power * pilot_power).clamp_min(1e-9))
                similarity = torch.sum(group_similarity * score_weight, dim=1) / torch.sum(
                    score_weight, dim=1
                ).clamp_min(1e-9)
            local_code_index = similarity.argmax(dim=1)
            if self.identity_mode == "local":
                selected_code_index = local_code_index
            else:
                confidence = torch.abs(similarity[:, 0] - similarity[:, 1])
                centered_bin_position = torch.where(
                    own_code_index == 1,
                    identity_threshold - snr_db - 0.625,
                    torch.zeros_like(snr_db),
                )
                adaptive_margin = (
                    self.identity_margin
                    + self.identity_margin_snr_slope * snr_db
                    + self.identity_margin_bin_slope * centered_bin_position
                ).clamp(0.0, 1.0)
                use_local = confidence >= adaptive_margin
                if self.identity_mode == "safe_hybrid":
                    use_local = use_local & (own_code_index == 1) & (local_code_index == 0)
                elif self.identity_mode == "window_hybrid":
                    use_local = (
                        use_local
                        & (own_code_index == 1)
                        & (local_code_index == 0)
                        & ((identity_threshold - snr_db) <= self.identity_window)
                    )
                elif self.identity_mode == "asymmetric_hybrid":
                    margin = torch.where(
                        own_code_index == 1,
                        torch.full_like(confidence, self.identity_margin_to_zero),
                        torch.full_like(confidence, self.identity_margin_to_one),
                    )
                    use_local = confidence >= margin
                selected_code_index = torch.where(use_local, local_code_index, own_code_index)
        own_gather = selected_code_index[:, None, None, None].expand(-1, 12, 1, 2)
        other_gather = (1 - selected_code_index)[:, None, None, None].expand(-1, 12, 1, 2)
        desired_vector = torch.gather(vectors, 2, own_gather).squeeze(2)
        other_vector = torch.gather(vectors, 2, other_gather).squeeze(2)
        pilot_bit_llr = None
        pilot_slope_llr = None
        if self.pilot_bit_min_snr < 90.0:
            assert local_vector is not None
            group_phase_statistic = torch.sum(
                local_vector.conj() * desired_vector, dim=2
            )
            if self.pilot_slope_bits:
                slope_values, slope_labels = _pilot_slope_constellation(
                    self.pilot_slope_bits,
                    self.pilot_slope_step,
                    device=received.device,
                )
                centered_groups = torch.arange(
                    12, device=received.device, dtype=received.real.dtype
                ) - 5.5
                slope_patterns = torch.polar(
                    torch.ones(
                        (len(slope_values), 12),
                        device=received.device,
                        dtype=received.real.dtype,
                    ),
                    slope_values[:, None] * centered_groups[None],
                ).to(received.dtype)
                slope_correlations = torch.sum(
                    group_phase_statistic[:, None] * slope_patterns[None].conj(), dim=2
                )
                slope_scores = torch.abs(slope_correlations)
                slope_index = slope_scores.argmax(dim=1)
                candidate_slope_pattern = slope_patterns[slope_index]
                slope_mask = snr_db >= self.pilot_slope_min_snr
                selected_slope_pattern = torch.where(
                    slope_mask[:, None],
                    candidate_slope_pattern,
                    torch.ones_like(candidate_slope_pattern),
                )
                desired_vector = desired_vector * selected_slope_pattern.conj()[..., None]
                group_phase_statistic = group_phase_statistic * selected_slope_pattern.conj()
                pilot_slope_llr = torch.stack(
                    [
                        slope_scores[:, slope_labels[:, bit_index]].amax(dim=1)
                        - slope_scores[:, ~slope_labels[:, bit_index]].amax(dim=1)
                        for bit_index in range(self.pilot_slope_bits)
                    ],
                    dim=1,
                )
            group_phase_magnitude = torch.abs(group_phase_statistic).clamp_min(1e-9)
            groups_per_phase_segment = 12 // self.pilot_phase_segments
            pilot_phase_statistic = torch.sum(
                (
                    group_phase_statistic
                    * group_phase_magnitude.pow(self.pilot_phase_weight_power - 1.0)
                ).reshape(batch, self.pilot_phase_segments, groups_per_phase_segment),
                dim=2,
            )
            pilot_phase_statistic = pilot_phase_statistic.reshape(batch, self.pilot_phase_segments)
            pilot_phase = torch.ones_like(pilot_phase_statistic)
            pilot_bit_llr = torch.zeros(
                (batch, self.pilot_phase_bits * self.pilot_phase_segments),
                device=received.device,
                dtype=received.real.dtype,
            )
            for phase_threshold, phase_bits in self.pilot_phase_schedule:
                phase_points, phase_labels = _pilot_phase_constellation(
                    phase_bits,
                    device=received.device,
                    dtype=received.dtype,
                )
                phase_scores = torch.real(
                    pilot_phase_statistic[..., None] * phase_points[None, None].conj()
                )
                phase_index = phase_scores.argmax(dim=2)
                candidate_phase = phase_points[phase_index]
                phase_mask = snr_db >= phase_threshold
                pilot_phase = torch.where(phase_mask[:, None], candidate_phase, pilot_phase)
                candidate_llr = torch.stack(
                    [
                        phase_scores[..., phase_labels[:, bit_index]].amax(dim=2)
                        - phase_scores[..., ~phase_labels[:, bit_index]].amax(dim=2)
                        for bit_index in range(phase_bits)
                    ],
                    dim=2,
                ).reshape(batch, phase_bits * self.pilot_phase_segments)
                active_phase_bits = phase_bits * self.pilot_phase_segments
                pilot_bit_llr[:, :active_phase_bits] = torch.where(
                    phase_mask[:, None],
                    candidate_llr,
                    pilot_bit_llr[:, :active_phase_bits],
                )
            expanded_phase = pilot_phase.repeat_interleave(groups_per_phase_segment, dim=1)
            desired_vector = desired_vector * expanded_phase.conj()[..., None]
        if self.steering_shrinkage > 0.0:
            assert local_vector is not None
            alignment = torch.sum(local_vector.conj() * desired_vector, dim=-1, keepdim=True) / torch.sum(
                torch.abs(local_vector).square(), dim=-1, keepdim=True
            ).clamp_min(1e-9)
            aligned_local = local_vector * alignment
            desired_vector = (1.0 - self.steering_shrinkage) * desired_vector + self.steering_shrinkage * aligned_local

        data_offsets = torch.tensor(
            [offset for offset in range(12) if offset != self.pilot_offset],
            device=received.device,
        )
        if self.frequency_interpolation == "linear":
            delta = data_offsets.to(received.real.dtype) - self.pilot_offset
            interpolation = torch.abs(delta) / 12.0

            def interpolate(values: torch.Tensor) -> torch.Tensor:
                previous = torch.roll(values, 1, dims=1)
                following = torch.roll(values, -1, dims=1)
                neighbor = torch.where(
                    (delta < 0)[None, None, :, None],
                    previous[:, :, None, :],
                    following[:, :, None, :],
                )
                return (
                    (1.0 - interpolation)[None, None, :, None] * values[:, :, None, :]
                    + interpolation[None, None, :, None] * neighbor
                )

            desired_for_filter = interpolate(desired_vector)
            other_for_filter = interpolate(other_vector)
        else:
            desired_for_filter = desired_vector
            other_for_filter = other_vector
        matrix = torch.stack([desired_for_filter, other_for_filter], dim=-1)
        covariance = matrix @ matrix.conj().transpose(-2, -1)
        if self.sample_covariance_blend > 0.0:
            data_offsets = torch.tensor(
                [offset for offset in range(12) if offset != self.pilot_offset],
                device=received.device,
            )
            grouped_data = grouped.index_select(-1, data_offsets).permute(0, 2, 1, 3)
            sample_covariance = torch.einsum(
                "bgrl,bgql->bgrq", grouped_data, grouped_data.conj()
            ) / grouped_data.shape[-1]
            covariance = (
                (1.0 - self.sample_covariance_blend) * covariance
                + self.sample_covariance_blend * sample_covariance
            )
        noise_variance = torch.pow(10.0, -snr_db / 10.0)
        pilot_estimation_noise = filter_energy * noise_variance / self.pilot_amplitude**2
        identity = torch.eye(2, device=received.device, dtype=received.dtype)
        loading = self.covariance_loading_scale * (noise_variance + pilot_estimation_noise)
        if self.frequency_interpolation == "linear":
            covariance = covariance + loading[:, None, None, None, None] * identity
            weights = torch.linalg.solve(covariance, desired_for_filter.unsqueeze(-1)).squeeze(-1)
            estimated_vectors = torch.stack([desired_for_filter, other_for_filter], dim=3)
            projected = torch.einsum("bglr,bglur->bglu", weights.conj(), estimated_vectors)
            data_y = grouped.index_select(-1, data_offsets).permute(0, 2, 3, 1)
            observation = torch.sum(weights.conj() * data_y, dim=-1).reshape(batch, 132)
            group_gain = projected[..., 0]
            filtered_noise = noise_variance[:, None, None] * torch.sum(torch.abs(weights).square(), dim=-1)
            residual = (torch.abs(projected[..., 1]).square() + filtered_noise).reshape(batch, 132)
        else:
            covariance = covariance + loading[:, None, None, None] * identity
            weights = torch.linalg.solve(covariance, desired_vector.unsqueeze(-1)).squeeze(-1)
            estimated_vectors = torch.stack([desired_vector, other_vector], dim=2)
            projected = torch.einsum("bgr,bgur->bgu", weights.conj(), estimated_vectors)
            data_y = grouped.index_select(-1, data_offsets).reshape(batch, 2, 132)
            weight_sc = weights.repeat_interleave(11, dim=1).permute(0, 2, 1)
            observation = torch.sum(weight_sc.conj() * data_y, dim=1)
            gain_interpolation_delta = None
            gain_post_modes = {
                "gain_linear_post",
                "gain_phase_post",
                "gain_magnitude_post",
                "gain_polar_post",
            }
            if self.frequency_interpolation == "gain_linear" or self.frequency_interpolation in gain_post_modes:
                center_gain = projected[:, :, 0]
                delta = data_offsets.to(received.real.dtype) - self.pilot_offset
                interpolation = torch.abs(delta) / 12.0
                neighbor_gain = torch.where(
                    (delta < 0)[None, None, :],
                    torch.roll(center_gain, 1, dims=1)[:, :, None],
                    torch.roll(center_gain, -1, dims=1)[:, :, None],
                )
                if self.frequency_interpolation == "gain_phase_post":
                    phase_delta = torch.angle(neighbor_gain * center_gain[:, :, None].conj())
                    interpolated_gain = center_gain[:, :, None] * torch.exp(
                        1j * interpolation[None, None, :] * phase_delta
                    )
                elif self.frequency_interpolation == "gain_magnitude_post":
                    magnitude = (
                        (1.0 - interpolation)[None, None, :] * torch.abs(center_gain)[:, :, None]
                        + interpolation[None, None, :] * torch.abs(neighbor_gain)
                    )
                    interpolated_gain = torch.polar(magnitude, torch.angle(center_gain)[:, :, None])
                elif self.frequency_interpolation == "gain_polar_post":
                    magnitude = (
                        (1.0 - interpolation)[None, None, :] * torch.abs(center_gain)[:, :, None]
                        + interpolation[None, None, :] * torch.abs(neighbor_gain)
                    )
                    phase_delta = torch.angle(neighbor_gain * center_gain[:, :, None].conj())
                    phase = torch.angle(center_gain)[:, :, None] + interpolation[None, None, :] * phase_delta
                    interpolated_gain = torch.polar(magnitude, phase)
                else:
                    interpolated_gain = (
                        (1.0 - interpolation)[None, None, :] * center_gain[:, :, None]
                        + interpolation[None, None, :] * neighbor_gain
                    )
                adaptive_gain_interpolation_scale = (
                    self.gain_interpolation_scale + self.gain_interpolation_snr_slope * snr_db
                ).clamp(0.0, 1.0)[:, None, None]
                gain_interpolation_delta = adaptive_gain_interpolation_scale * (
                    interpolated_gain - center_gain[:, :, None]
                )
                if self.frequency_interpolation == "gain_linear":
                    group_gain = center_gain[:, :, None] + gain_interpolation_delta
                else:
                    group_gain = center_gain[:, :, None]
            else:
                group_gain = projected[:, :, 0, None]
            filtered_noise = noise_variance[:, None] * torch.sum(torch.abs(weights).square(), dim=-1)
            residual = (torch.abs(projected[:, :, 1]).square() + filtered_noise).repeat_interleave(11, dim=1)
        group_observation = observation.reshape(batch, 12, 11)
        pre_vector_gain = projected[:, :, 0, None]
        if self.pre_vector_gain_refinement_iterations > 0:
            points, _ = qam_constellation(8, device=received.device, central_boost=self.central_boost)
            pre_update_mask = (snr_db >= self.gain_refinement_min_snr)[:, None, None]
            for _ in range(self.pre_vector_gain_refinement_iterations):
                safe_pre_gain = torch.where(
                    torch.abs(pre_vector_gain) > 1e-9,
                    pre_vector_gain,
                    torch.ones_like(pre_vector_gain),
                )
                pre_equalized = group_observation / safe_pre_gain
                pre_nearest = torch.argmin(
                    torch.abs(pre_equalized[..., None] - points).square(), dim=-1
                )
                pre_decisions = points[pre_nearest]
                pre_estimate = torch.sum(
                    group_observation * pre_decisions.conj(), dim=-1, keepdim=True
                ) / torch.sum(torch.abs(pre_decisions).square(), dim=-1, keepdim=True).clamp_min(1e-6)
                pre_updated = (
                    (1.0 - self.gain_refinement_rate) * pre_vector_gain
                    + self.gain_refinement_rate * pre_estimate
                )
                pre_vector_gain = torch.where(pre_update_mask, pre_updated, pre_vector_gain)
        if self.data_vector_refinement_scale > 0.0 and self.frequency_interpolation != "linear":
            points, _ = qam_constellation(8, device=received.device, central_boost=self.central_boost)
            grouped_data_y = data_y.reshape(batch, 2, 12, 11).permute(0, 2, 1, 3)
            for vector_iteration in range(self.data_vector_refinement_iterations):
                initial_gain = pre_vector_gain if vector_iteration == 0 else projected[:, :, 0, None]
                safe_gain = torch.where(
                    torch.abs(initial_gain) > 1e-9,
                    initial_gain,
                    torch.ones_like(initial_gain),
                )
                equalized = group_observation / safe_gain
                if self.data_vector_soft_temperature > 0.0:
                    vector_group_residual = (
                        torch.abs(projected[:, :, 1]).square()
                        + noise_variance[:, None] * torch.sum(torch.abs(weights).square(), dim=-1)
                    )
                    predicted = initial_gain[..., None] * points
                    normalized_distance = torch.abs(
                        group_observation[..., None] - predicted
                    ).square() / vector_group_residual[:, :, None, None].clamp_min(1e-6)
                    posterior = torch.softmax(
                        -normalized_distance / self.data_vector_soft_temperature, dim=-1
                    )
                    soft_conjugate = torch.sum(posterior * points.conj(), dim=-1)
                    soft_energy = torch.sum(posterior * torch.abs(points).square(), dim=-1)
                    vector_confidence = posterior.amax(dim=-1).mean(dim=-1, keepdim=True)
                    reliability = posterior.amax(dim=-1).pow(self.data_vector_reliability_power)
                    vector_estimate = torch.sum(
                        grouped_data_y * (soft_conjugate * reliability)[:, :, None, :], dim=-1
                    ) / torch.sum(soft_energy * reliability, dim=-1, keepdim=True).clamp_min(1e-6)
                else:
                    nearest = torch.argmin(
                        torch.abs(equalized[..., None] - points).square(), dim=-1
                    )
                    decisions = points[nearest]
                    vector_estimate = torch.sum(
                        grouped_data_y * decisions.conj()[:, :, None, :], dim=-1
                    ) / torch.sum(torch.abs(decisions).square(), dim=-1, keepdim=True).clamp_min(1e-6)
                vector_correction = vector_estimate - desired_vector
                if self.data_vector_smoothing_side > 0.0:
                    vector_correction = (
                        self.data_vector_smoothing_side * torch.roll(vector_correction, 1, dims=1)
                        + (1.0 - 2.0 * self.data_vector_smoothing_side) * vector_correction
                        + self.data_vector_smoothing_side * torch.roll(vector_correction, -1, dims=1)
                    )
                adaptive_vector_scale = (
                    self.data_vector_refinement_scale + self.data_vector_refinement_snr_slope * snr_db
                ).clamp(0.0, 1.0)[:, None, None]
                if self.data_vector_soft_temperature > 0.0:
                    confidence_scale = self.data_vector_confidence_floor + (
                        1.0 - self.data_vector_confidence_floor
                    ) * vector_confidence
                    adaptive_vector_scale = adaptive_vector_scale * confidence_scale
                refined_vector = desired_vector + adaptive_vector_scale * vector_correction
                vector_update_mask = (snr_db >= self.data_vector_refinement_min_snr)[:, None, None]
                desired_vector = torch.where(vector_update_mask, refined_vector, desired_vector)
                matrix = torch.stack([desired_vector, other_vector], dim=-1)
                covariance = matrix @ matrix.conj().transpose(-2, -1)
                covariance = covariance + loading[:, None, None, None] * identity
                weights = torch.linalg.solve(covariance, desired_vector.unsqueeze(-1)).squeeze(-1)
                estimated_vectors = torch.stack([desired_vector, other_vector], dim=2)
                projected = torch.einsum("bgr,bgur->bgu", weights.conj(), estimated_vectors)
                weight_sc = weights.repeat_interleave(11, dim=1).permute(0, 2, 1)
                observation = torch.sum(weight_sc.conj() * data_y, dim=1)
                group_observation = observation.reshape(batch, 12, 11)
            if self.frequency_interpolation in gain_post_modes:
                center_gain = projected[:, :, 0]
                neighbor_gain = torch.where(
                    (delta < 0)[None, None, :],
                    torch.roll(center_gain, 1, dims=1)[:, :, None],
                    torch.roll(center_gain, -1, dims=1)[:, :, None],
                )
                interpolated_gain = (
                    (1.0 - interpolation)[None, None, :] * center_gain[:, :, None]
                    + interpolation[None, None, :] * neighbor_gain
                )
                gain_interpolation_delta = adaptive_gain_interpolation_scale * (
                    interpolated_gain - center_gain[:, :, None]
                )
        if self.gain_refinement_iterations:
            points, _ = qam_constellation(8, device=received.device, central_boost=self.central_boost)
            update_mask = (snr_db >= self.gain_refinement_min_snr)[:, None, None]
            group_residual = torch.abs(projected[:, :, 1]).square() + noise_variance[:, None] * torch.sum(
                torch.abs(weights).square(), dim=-1
            )
            for _ in range(self.gain_refinement_iterations):
                if self.gain_refinement_soft_temperature > 0.0:
                    predicted = group_gain[..., None] * points
                    normalized_distance = torch.abs(
                        group_observation[..., None] - predicted
                    ).square() / group_residual[:, :, None, None].clamp_min(1e-6)
                    posterior = torch.softmax(
                        -normalized_distance / self.gain_refinement_soft_temperature, dim=-1
                    )
                    soft_conjugate = torch.sum(posterior * points.conj(), dim=-1)
                    soft_energy = torch.sum(posterior * torch.abs(points).square(), dim=-1)
                    estimate = torch.sum(
                        group_observation * soft_conjugate, dim=-1, keepdim=True
                    ) / torch.sum(soft_energy, dim=-1, keepdim=True).clamp_min(1e-6)
                else:
                    safe_gain = torch.where(
                        torch.abs(group_gain) > 1e-9, group_gain, torch.ones_like(group_gain)
                    )
                    equalized = group_observation / safe_gain
                    nearest = torch.argmin(
                        torch.abs(equalized[..., None] - points).square(), dim=-1
                    )
                    decisions = points[nearest]
                    estimate = torch.sum(
                        group_observation * decisions.conj(), dim=-1, keepdim=True
                    ) / torch.sum(torch.abs(decisions).square(), dim=-1, keepdim=True).clamp_min(1e-6)
                updated = (1.0 - self.gain_refinement_rate) * group_gain + self.gain_refinement_rate * estimate
                group_gain = torch.where(update_mask, updated, group_gain)
        if self.frequency_interpolation in {
            "gain_linear_post",
            "gain_phase_post",
            "gain_magnitude_post",
            "gain_polar_post",
        }:
            assert gain_interpolation_delta is not None
            group_gain = group_gain + gain_interpolation_delta
        if self.data_gain_refinement_scale > 0.0:
            points, _ = qam_constellation(8, device=received.device, central_boost=self.central_boost)
            flat_observation = group_observation.reshape(batch, 132)
            if self.data_gain_residual_mode == "current":
                data_gain_residual = (
                    torch.abs(projected[:, :, 1]).square()
                    + noise_variance[:, None] * torch.sum(torch.abs(weights).square(), dim=-1)
                ).repeat_interleave(11, dim=1)
            else:
                data_gain_residual = residual
            shifts = list(range(-self.data_gain_refinement_radius, self.data_gain_refinement_radius + 1))
            if self.data_gain_kernel == "binom5":
                if self.data_gain_refinement_radius != 2:
                    raise ValueError("binom5 data gain kernel requires radius 2")
                kernel_weights = (1.0, 4.0, 6.0, 4.0, 1.0)
            elif self.data_gain_kernel == "triangular":
                kernel_weights = tuple(
                    self.data_gain_refinement_radius + 1 - abs(shift) for shift in shifts
                )
            else:
                kernel_weights = tuple(1.0 for _ in shifts)
            kernel_weights = tuple(
                self.data_gain_center_weight if shift == 0 else weight
                for weight, shift in zip(kernel_weights, shifts, strict=True)
            )
            base_gain = group_gain.reshape(batch, 132)
            data_positions = (
                12 * torch.arange(12, device=received.device)[:, None]
                + data_offsets[None, :]
            ).reshape(-1)
            physical_deltas = []
            for shift in shifts:
                delta_position = torch.roll(data_positions, shift) - data_positions
                physical_deltas.append(torch.remainder(delta_position + 72, 144) - 72)
            update_mask = (snr_db >= self.data_gain_refinement_min_snr)[:, None]
            adaptive_refinement_scale = (
                self.data_gain_refinement_scale + self.data_gain_refinement_snr_slope * snr_db
            ).clamp(0.0, 1.0)[:, None]
            for iteration in range(self.data_gain_refinement_iterations):
                safe_gain = torch.where(
                    torch.abs(base_gain) > 1e-9, base_gain, torch.ones_like(base_gain)
                ).reshape(batch, 12, 11)
                equalized = group_observation / safe_gain
                if self.data_gain_soft_temperature > 0.0:
                    predicted = base_gain[..., None] * points
                    normalized_distance = torch.abs(
                        flat_observation[..., None] - predicted
                    ).square() / data_gain_residual[..., None].clamp_min(1e-6)
                    posterior = torch.softmax(
                        -normalized_distance
                        / (
                            self.data_gain_soft_temperature
                            * self.data_gain_soft_temperature_decay**iteration
                        ),
                        dim=-1,
                    )
                    soft_conjugate = torch.sum(posterior * points.conj(), dim=-1)
                    numerator = flat_observation * soft_conjugate
                    denominator = torch.sum(posterior * torch.abs(points).square(), dim=-1)
                else:
                    nearest = torch.argmin(
                        torch.abs(equalized[..., None] - points).square(), dim=-1
                    )
                    flat_decisions = points[nearest].reshape(batch, 132)
                    numerator = flat_observation * flat_decisions.conj()
                    denominator = torch.abs(flat_decisions).square()
                rolled_numerators = [
                    weight * torch.roll(numerator, shift, dims=1)
                    for weight, shift in zip(kernel_weights, shifts, strict=True)
                ]
                rolled_denominators = [
                    weight * torch.roll(denominator, shift, dims=1)
                    for weight, shift in zip(kernel_weights, shifts, strict=True)
                ]
                smoothed_numerator = sum(rolled_numerators)
                smoothed_denominator = sum(rolled_denominators)
                if self.data_gain_model == "linear":
                    first_numerator = sum(
                        value * delta[None]
                        for value, delta in zip(rolled_numerators, physical_deltas, strict=True)
                    )
                    first_denominator = sum(
                        value * delta[None]
                        for value, delta in zip(rolled_denominators, physical_deltas, strict=True)
                    )
                    second_denominator = sum(
                        value * delta[None].square()
                        for value, delta in zip(rolled_denominators, physical_deltas, strict=True)
                    )
                    determinant = (
                        smoothed_denominator * second_denominator - first_denominator.square()
                    ).clamp_min(1e-6)
                    smoothed_gain = (
                        smoothed_numerator * second_denominator
                        - first_numerator * first_denominator
                    ) / determinant
                else:
                    smoothed_gain = smoothed_numerator / smoothed_denominator.clamp_min(1e-6)
                refined_gain = base_gain + adaptive_refinement_scale * (smoothed_gain - base_gain)
                base_gain = torch.where(update_mask, refined_gain, base_gain)
            group_gain = base_gain
        if group_gain.shape[-1] == 1:
            desired_gain = group_gain.repeat_interleave(11, dim=1).reshape(batch, 132)
        else:
            desired_gain = group_gain.reshape(batch, 132)
        joint_llr = None
        if self.interference_cancellation_scale > 0.0 and self.frequency_interpolation != "linear":
            points, labels = qam_constellation(
                8, device=received.device, central_boost=self.central_boost
            )
            interference_covariance = matrix @ matrix.conj().transpose(-2, -1)
            interference_covariance = (
                interference_covariance
                + self.interference_filter_loading_scale * loading[:, None, None, None] * identity
            )
            other_weights = torch.linalg.solve(
                interference_covariance, other_vector.unsqueeze(-1)
            ).squeeze(-1)
            other_projected = torch.einsum("bgr,bgur->bgu", other_weights.conj(), estimated_vectors)
            other_weight_sc = other_weights.repeat_interleave(11, dim=1).permute(0, 2, 1)
            other_observation = torch.sum(other_weight_sc.conj() * data_y, dim=1)
            other_gain = other_projected[:, :, 1].repeat_interleave(11, dim=1)
            desired_distance = torch.abs(
                observation[..., None] - desired_gain[..., None] * points
            ).square() / residual[..., None].clamp_min(1e-6)
            desired_posterior = torch.softmax(
                -desired_distance / self.reciprocal_cancellation_temperature,
                dim=-1,
            )
            soft_desired = torch.sum(desired_posterior * points, dim=-1)
            desired_leakage_into_other = other_projected[:, :, 0].repeat_interleave(11, dim=1)
            other_observation = (
                other_observation
                - self.reciprocal_cancellation_scale
                * desired_leakage_into_other
                * soft_desired
            )
            other_residual = (
                torch.abs(other_projected[:, :, 0]).square()
                + noise_variance[:, None] * torch.sum(torch.abs(other_weights).square(), dim=-1)
            ).repeat_interleave(11, dim=1)
            for _ in range(self.interference_gain_refinement_iterations):
                other_distance = torch.abs(
                    other_observation[..., None] - other_gain[..., None] * points
                ).square() / other_residual[..., None].clamp_min(1e-6)
                other_posterior = torch.softmax(
                    -other_distance / self.interference_cancellation_temperature,
                    dim=-1,
                )
                other_soft_conjugate = torch.sum(other_posterior * points.conj(), dim=-1)
                other_soft_energy = torch.sum(other_posterior * torch.abs(points).square(), dim=-1)
                other_numerator = other_observation * other_soft_conjugate
                other_smoothed_numerator = sum(
                    torch.roll(other_numerator, shift, dims=1) for shift in range(-2, 3)
                )
                other_smoothed_denominator = sum(
                    torch.roll(other_soft_energy, shift, dims=1) for shift in range(-2, 3)
                )
                other_gain_estimate = (
                    other_smoothed_numerator / other_smoothed_denominator.clamp_min(1e-6)
                )
                other_gain = other_gain + self.interference_gain_refinement_scale * (
                    other_gain_estimate - other_gain
                )
            other_distance = torch.abs(
                other_observation[..., None] - other_gain[..., None] * points
            ).square() / other_residual[..., None].clamp_min(1e-6)
            other_posterior = torch.softmax(
                -other_distance / self.interference_cancellation_temperature,
                dim=-1,
            )
            soft_other = torch.sum(other_posterior * points, dim=-1)
            other_confidence = other_posterior.amax(dim=-1)
            cancellation_confidence = self.interference_cancellation_confidence_floor + (
                1.0 - self.interference_cancellation_confidence_floor
            ) * other_confidence
            other_soft_energy = torch.sum(other_posterior * torch.abs(points).square(), dim=-1)
            interference_grouped_data_y = data_y.reshape(batch, 2, 12, 11).permute(0, 2, 1, 3)
            other_vector_estimate = torch.sum(
                interference_grouped_data_y * soft_other.conj().reshape(batch, 12, 1, 11),
                dim=-1,
            ) / torch.sum(
                other_soft_energy.reshape(batch, 12, 11), dim=-1, keepdim=True
            ).clamp_min(1e-6)
            refined_other_vector = other_vector + self.interference_vector_refinement_scale * (
                other_vector_estimate - other_vector
            )
            other_leakage = torch.einsum(
                "bgr,bgr->bg", weights.conj(), refined_other_vector
            ).repeat_interleave(11, dim=1)
            adaptive_cancellation_scale = (
                self.interference_cancellation_scale
                + self.interference_cancellation_snr_slope * snr_db
            ).clamp(0.0, 1.0)[:, None]
            cancelled = (
                observation
                - adaptive_cancellation_scale
                * cancellation_confidence
                * other_leakage
                * soft_other
            )
            cancellation_mask = (snr_db >= self.interference_cancellation_min_snr)[:, None]
            observation = torch.where(cancellation_mask, cancelled, observation)
            if self.joint_detection_candidates > 0:
                candidate_indices = torch.topk(
                    other_distance,
                    self.joint_detection_candidates,
                    dim=-1,
                    largest=False,
                ).indices
                candidate_points = points[candidate_indices]
                candidate_prior = torch.gather(other_distance, -1, candidate_indices)
                desired_vector_sc = estimated_vectors[:, :, 0].repeat_interleave(11, dim=1)
                other_vector_sc = estimated_vectors[:, :, 1].repeat_interleave(11, dim=1)
                received_sc = data_y.permute(0, 2, 1)
                desired_statistic = torch.sum(
                    desired_vector_sc.conj() * received_sc, dim=-1
                )
                other_statistic = torch.sum(other_vector_sc.conj() * received_sc, dim=-1)
                desired_energy = torch.sum(torch.abs(desired_vector_sc).square(), dim=-1)
                other_energy = torch.sum(torch.abs(other_vector_sc).square(), dim=-1)
                vector_cross = torch.sum(
                    desired_vector_sc.conj() * other_vector_sc, dim=-1
                )
                desired_metric = (
                    desired_energy[..., None] * torch.abs(points).square()
                    - 2.0 * torch.real(points.conj() * desired_statistic[..., None])
                )
                other_metric = (
                    other_energy[..., None] * torch.abs(candidate_points).square()
                    - 2.0
                    * torch.real(candidate_points.conj() * other_statistic[..., None])
                )
                cross_metric = 2.0 * torch.real(
                    points[None, None, :, None].conj()
                    * vector_cross[..., None, None]
                    * candidate_points[..., None, :]
                )
                joint_distance = (
                    desired_metric[..., :, None] + other_metric[..., None, :] + cross_metric
                ) / noise_variance[:, None, None, None].clamp_min(1e-6)
                joint_distance = joint_distance + (
                    self.joint_detection_prior_scale * candidate_prior[..., None, :]
                )
                desired_distance = joint_distance.amin(dim=-1)
                joint_axis_llrs = []
                for bit_index in range(8):
                    bit_is_one = labels[:, bit_index] > 0.5
                    min_zero = desired_distance[..., ~bit_is_one].amin(dim=-1)
                    min_one = desired_distance[..., bit_is_one].amin(dim=-1)
                    joint_axis_llrs.append(min_zero - min_one)
                joint_axis_llr = torch.stack(joint_axis_llrs, dim=-1)
                joint_layers = torch.empty_like(joint_axis_llr)
                joint_layers[..., 0::2] = joint_axis_llr[..., :4]
                joint_layers[..., 1::2] = joint_axis_llr[..., 4:]
                joint_llr = joint_layers.transpose(-2, -1).reshape(batch, -1)
        standard_llr = layered_qam_maxlog_llr(
            observation,
            desired_gain,
            residual,
            8,
            central_boost=self.central_boost,
        )
        if joint_llr is not None:
            joint_mask = (snr_db >= self.joint_detection_min_snr)[:, None]
            standard_llr = torch.where(joint_mask, joint_llr, standard_llr)
        if pilot_bit_llr is not None:
            standard_llr = torch.cat([standard_llr, pilot_bit_llr], dim=1)
        if pilot_slope_llr is not None:
            standard_llr = torch.cat([standard_llr, pilot_slope_llr], dim=1)
        return standard_llr, selected_code_index

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
        precoder_group = rzf_precoder(
            decoded,
            torch.pow(10.0, -snr_dl / 10.0),
            self.regularization_scale,
            self.fairness_exponent,
        )
        precoder_sc = precoder_group.repeat_interleave(12, dim=1)

        streams = torch.zeros((batch, 2, 144), device=channel.device, dtype=channel.dtype)
        data_symbols = torch.stack(
            [layered_qam_modulate(bits[:, user, :1056], 8, central_boost=self.central_boost) for user in range(2)],
            dim=1,
        )
        data_offsets = torch.tensor(
            [offset for offset in range(12) if offset != self.pilot_offset],
            device=channel.device,
        )
        data_positions = (12 * torch.arange(12, device=channel.device)[:, None] + data_offsets[None]).reshape(-1)
        streams[:, :, data_positions] = data_symbols
        actual_code_index, inferred_code_index, threshold_index = _pilot_assignment(snr_dl)
        identity_threshold = -20.0 + 1.25 * threshold_index.to(torch.float32)
        if self.control_levels != 31:
            lower_snr = snr_dl.min(dim=1).values
            normalized = ((lower_snr + 20.0) / 40.0).clamp(0.0, 1.0 - 1e-7)
            threshold_index = torch.floor(
                normalized.pow(self.control_companding) * self.control_levels
            ).to(torch.int64)
            upper_normalized = ((threshold_index + 1).to(torch.float32) / self.control_levels).pow(
                1.0 / self.control_companding
            )
            threshold = -20.0 + 40.0 * upper_normalized
            inferred_code_index = (snr_dl <= threshold[:, None]).to(torch.int64)
            identity_threshold = threshold
        pilot_positions = 12 * torch.arange(12, device=channel.device) + self.pilot_offset
        walsh_sign = torch.where(
            torch.arange(12, device=channel.device) % 2 == 0,
            torch.ones(12, device=channel.device),
            -torch.ones(12, device=channel.device),
        ).to(channel.dtype)
        for user in range(2):
            code = torch.where(
                (actual_code_index[:, user] == 0)[:, None],
                torch.ones((batch, 12), device=channel.device, dtype=channel.dtype),
                walsh_sign[None],
            )
            if self.pilot_bit_min_snr < 90.0:
                pilot_phase = torch.ones(
                    (batch, self.pilot_phase_segments),
                    device=channel.device,
                    dtype=channel.dtype,
                )
                for phase_threshold, phase_bits in self.pilot_phase_schedule:
                    phase_points, _ = _pilot_phase_constellation(
                        phase_bits,
                        device=channel.device,
                        dtype=channel.dtype,
                    )
                    active_phase_bits = phase_bits * self.pilot_phase_segments
                    pilot_bits = bits[
                        :, user, 1056 : 1056 + active_phase_bits
                    ].reshape(batch, self.pilot_phase_segments, phase_bits)
                    bit_weights = 2 ** torch.arange(
                        phase_bits - 1,
                        -1,
                        -1,
                        device=channel.device,
                        dtype=torch.int64,
                    )
                    phase_index = torch.sum(
                        pilot_bits.to(torch.int64) * bit_weights[None, None], dim=2
                    )
                    candidate_phase = phase_points[phase_index]
                    pilot_phase = torch.where(
                        (snr_dl[:, user] >= phase_threshold)[:, None],
                        candidate_phase,
                        pilot_phase,
                    )
                expanded_phase = pilot_phase.repeat_interleave(
                    12 // self.pilot_phase_segments, dim=1
                )
                code = code * expanded_phase
                if self.pilot_slope_bits:
                    slope_values, _ = _pilot_slope_constellation(
                        self.pilot_slope_bits,
                        self.pilot_slope_step,
                        device=channel.device,
                    )
                    slope_bits = bits[
                        :,
                        user,
                        1056 + self.pilot_phase_bits :
                        1056 + self.pilot_phase_bits + self.pilot_slope_bits,
                    ]
                    slope_weights = 2 ** torch.arange(
                        self.pilot_slope_bits - 1,
                        -1,
                        -1,
                        device=channel.device,
                        dtype=torch.int64,
                    )
                    slope_index = torch.sum(
                        slope_bits.to(torch.int64) * slope_weights[None], dim=1
                    )
                    centered_groups = torch.arange(
                        12, device=channel.device, dtype=channel.real.dtype
                    ) - 5.5
                    candidate_slope_pattern = torch.polar(
                        torch.ones((batch, 12), device=channel.device, dtype=channel.real.dtype),
                        slope_values[slope_index, None] * centered_groups[None],
                    ).to(channel.dtype)
                    slope_pattern = torch.where(
                        (snr_dl[:, user] >= self.pilot_slope_min_snr)[:, None],
                        candidate_slope_pattern,
                        torch.ones_like(candidate_slope_pattern),
                    )
                    code = code * slope_pattern
            streams[:, user, pilot_positions] = self.pilot_amplitude * code

        signal = torch.einsum("bstu,bus->bts", precoder_sc, streams)
        signal, _ = normalize_downlink(signal)
        received, _ = add_awgn(apply_downlink(channel, signal), snr_dl, generator=generator)
        received_users = [
            self._receive(
                received[:, user],
                channel[:, user],
                snr_dl[:, user],
                inferred_code_index[:, user],
                identity_threshold,
            )
            for user in range(2)
        ]
        llr = torch.stack([result[0] for result in received_users], dim=1)
        receiver_code_index = torch.stack([result[1] for result in received_users], dim=1)
        identity_accuracy = (receiver_code_index == actual_code_index).to(torch.float32)
        return llr, identity_accuracy

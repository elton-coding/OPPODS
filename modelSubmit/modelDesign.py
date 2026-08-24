from __future__ import annotations

import math

import torch
from torch import nn

NUM_UE = 2
NUM_UL_RE = 96
NUM_DL_SC = 144
NUM_TX = 16
NUM_CTRL = 5
GROUP_SIZE = 12
GROUPS = 12
MODE_COUNT = 6
DELAY_MODE_ORDER = (0, 1, 2, 3, 11, 10)
DELAY_MODE_PRIOR = (0.5722, 0.2068, 0.0698, 0.0301, 0.0424, 0.0173)
WIENER_NOISE_SCALE = 0.75
RZF_REGULARIZATION = 1.5
PILOT_RZF_REGULARIZATION = 0.45
CENTRAL_BOOST = -0.5
DD_ITERATIONS = 15
LOW_SNR_THRESHOLD_DB = -20.0
MIDDLE_PREFIX_THRESHOLD_DB = -8.0
MIDDLE_PREFIX_BITS = 924
RESERVED_PROFILE_MAX_SNR_DB = 20.0
PILOT_AMPLITUDE = 1.5
PILOT_OFFSET = 5
PILOT_COVARIANCE_LOADING_SCALE = 4.0
PILOT_GAIN_REFINEMENT_ITERATIONS = 4
PILOT_GAIN_REFINEMENT_MIN_SNR_DB = -10.0
PILOT_GAIN_REFINEMENT_RATE = 0.3125
PILOT_IDENTITY_MARGIN = 0.3
PILOT_IDENTITY_MARGIN_SNR_SLOPE = -0.02
PILOT_GAIN_INTERPOLATION_SCALE = 0.5
PILOT_GAIN_INTERPOLATION_SCALE_INTERVALS = (
    (-17.5, -15.0, 0.75),
    (-10.0, -7.5, 0.75),
    (15.0, 20.0, 0.75),
)
PILOT_STEERING_SHRINKAGE = 0.0375
DATA_GAIN_REFINEMENT_SCALE = 0.2
DATA_GAIN_REFINEMENT_RADIUS = 2
DATA_GAIN_REFINEMENT_MIN_SNR_DB = -5.0
DATA_GAIN_REFINEMENT_ITERATIONS = 4
DATA_GAIN_REFINEMENT_SNR_SLOPE = 0.01
DATA_GAIN_SOFT_TEMPERATURE = 0.5
DATA_VECTOR_REFINEMENT_SCALE = 0.2
DATA_VECTOR_REFINEMENT_MIN_SNR_DB = -12.5
DATA_VECTOR_SOFT_TEMPERATURE = 3.0
INTERFERENCE_CANCELLATION_SCALE = 0.2
INTERFERENCE_CANCELLATION_TEMPERATURE = 0.3
INTERFERENCE_CANCELLATION_MIN_SNR_DB = -10.0
INTERFERENCE_CANCELLATION_SNR_SLOPE = -0.005
INTERFERENCE_CANCELLATION_DISABLED_INTERVALS_DB = ((-8.5, -8.0),)
PILOT_BIT_MIN_SNR_DB = 2.5
PILOT_8PSK_MIN_SNR_DB = 7.5
PILOT_16PSK_MIN_SNR_DB = 12.5
PILOT_32PSK_MIN_SNR_DB = 18.75
DATA_VECTOR_SOFT_TEMPERATURE_INTERVALS = ((0.0, 2.5, 2.0), (5.0, 7.5, 2.0))
DATA_GAIN_SOFT_TEMPERATURE_INTERVALS = ((-2.5, 0.0, 0.3), (2.5, 5.0, 0.3))


def _snr_interval_value(
    default: float,
    intervals: tuple[tuple[float, float, float], ...],
    snr: torch.Tensor,
) -> torch.Tensor:
    value = torch.full_like(snr, default)
    for low_db, high_db, interval_value in intervals:
        in_interval = (snr >= low_db) & (snr < high_db)
        value = torch.where(in_interval, torch.full_like(value, interval_value), value)
    return value


def _dominant_hermitian_eigenvector_2x2(matrix: torch.Tensor) -> torch.Tensor:
    a = matrix[..., 0, 0].real
    d = matrix[..., 1, 1].real
    c = matrix[..., 0, 1]
    discriminant = torch.sqrt((a - d).square() + 4.0 * torch.abs(c).square())
    largest = 0.5 * (a + d + discriminant)
    candidate_a = torch.stack([torch.complex(largest - d, torch.zeros_like(largest)), c.conj()], dim=-1)
    candidate_d = torch.stack([c, torch.complex(largest - a, torch.zeros_like(largest))], dim=-1)
    vector = torch.where((a >= d)[..., None], candidate_a, candidate_d)
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    fallback = torch.zeros_like(vector)
    fallback[..., 0] = 1.0
    return torch.where(norm > 1e-12, vector / norm.clamp_min(1e-12), fallback)


def _task_feedback(channel: torch.Tensor) -> torch.Tensor:
    batch, rx, tx, _ = channel.shape
    grouped = channel.reshape(batch, rx, tx, GROUPS, GROUP_SIZE).permute(0, 3, 1, 2, 4)
    covariance = torch.einsum("bgrtl,bgqtl->bgrq", grouped, grouped.conj())
    combiner = _dominant_hermitian_eigenvector_2x2(covariance)
    anchor_index = torch.abs(combiner).argmax(dim=-1, keepdim=True)
    anchor = torch.gather(combiner, -1, anchor_index)
    combiner = combiner * (anchor / torch.abs(anchor).clamp_min(1e-9)).conj()
    repeated = combiner.repeat_interleave(GROUP_SIZE, dim=1).permute(0, 2, 1)
    effective_sc = torch.sum(repeated.conj().unsqueeze(2) * channel, dim=1)
    return effective_sc.reshape(batch, tx, GROUPS, GROUP_SIZE).mean(dim=-1).permute(0, 2, 1)


def _normalize_feedback(signal: torch.Tensor) -> torch.Tensor:
    return signal / torch.sqrt(torch.mean(torch.abs(signal).square(), dim=-1, keepdim=True).clamp_min(1e-9))


def _pilot_assignment(snr_db: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    actual = torch.empty_like(snr_db, dtype=torch.int64)
    user0_is_high = snr_db[:, 0] >= snr_db[:, 1]
    actual[:, 0] = (~user0_is_high).to(torch.int64)
    actual[:, 1] = user0_is_high.to(torch.int64)
    lower_snr = snr_db.min(dim=1).values
    threshold_index = torch.floor((lower_snr + 20.0) / (30.25 / 27.0)).to(torch.int64).clamp(0, 26)
    return actual, threshold_index


def _integer_to_control(value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    shifts = torch.arange(4, -1, -1, device=value.device, dtype=torch.int64)
    return torch.bitwise_and(torch.bitwise_right_shift(value[:, None], shifts[None, :]), 1).to(dtype)


def _tail_codebook(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    positions = torch.arange(96, device=device, dtype=torch.int64)[None, None]
    roles = torch.arange(2, device=device, dtype=torch.int64)[None, :, None]
    bases = torch.arange(4, device=device, dtype=torch.int64)[:, None, None]
    hashed = positions + 97 * roles + 193 * bases + 1
    hashed = hashed * 1103515245 + 12345
    hashed = torch.bitwise_xor(hashed, torch.bitwise_right_shift(hashed, 13))
    hashed = hashed * 2654435761
    base = torch.bitwise_and(torch.bitwise_right_shift(hashed, 17), 1)
    full = torch.cat([base, 1 - base], dim=0)
    return full[torch.tensor((0, 1, 3, 5, 7), device=device)].to(dtype)


def _threshold_tail_codebook(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    positions = torch.arange(228, device=device, dtype=torch.int64)[None]
    templates = torch.arange(14, device=device, dtype=torch.int64)[:, None]
    hashed = positions + 193 * templates + 1
    hashed = hashed * 1103515245 + 12345
    hashed = torch.bitwise_xor(hashed, torch.bitwise_right_shift(hashed, 13))
    hashed = hashed * 2654435761
    base = torch.bitwise_and(torch.bitwise_right_shift(hashed, 17), 1)
    paired = torch.stack([base, 1 - base], dim=1).reshape(28, 228)
    return paired[:27].to(dtype)


def _bits_to_integer(bits: torch.Tensor) -> torch.Tensor:
    width = bits.shape[-1]
    weights = 2 ** torch.arange(width - 1, -1, -1, device=bits.device, dtype=torch.int64)
    return torch.sum(bits.to(torch.int64) * weights, dim=-1)


def _gray_to_binary(gray: torch.Tensor) -> torch.Tensor:
    binary = gray.clone()
    shifted = gray.clone()
    while torch.any(shifted > 0):
        shifted = torch.bitwise_right_shift(shifted, 1)
        binary = torch.bitwise_xor(binary, shifted)
    return binary


def _pilot_phase_constellation(
    bits: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels_int = torch.arange(2**bits, device=device, dtype=torch.int64)
    binary = _gray_to_binary(labels_int)
    angle = 2.0 * math.pi * binary.to(torch.float32) / (2**bits)
    points = torch.polar(torch.ones_like(angle), angle).to(dtype=dtype)
    shifts = torch.arange(bits - 1, -1, -1, device=device, dtype=torch.int64)
    labels = torch.bitwise_and(
        torch.bitwise_right_shift(labels_int[:, None], shifts[None]), 1
    ).to(torch.bool)
    return points, labels


def _decode_pilot_phase(
    statistic: torch.Tensor,
    bits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    points, labels = _pilot_phase_constellation(
        bits,
        device=statistic.device,
        dtype=statistic.dtype,
    )
    scores = torch.real(statistic[:, None] * points[None].conj())
    phase = points[scores.argmax(dim=1)]
    llr = torch.stack(
        [
            scores[:, labels[:, bit_index]].amax(dim=1)
            - scores[:, ~labels[:, bit_index]].amax(dim=1)
            for bit_index in range(bits)
        ],
        dim=1,
    )
    return phase, llr


def _qam_modulate(bits: torch.Tensor) -> torch.Tensor:
    grouped = bits.reshape(*bits.shape[:-1], -1, 8)
    i_index = _gray_to_binary(_bits_to_integer(grouped[..., :4]))
    q_index = _gray_to_binary(_bits_to_integer(grouped[..., 4:]))
    i_level = 2.0 * i_index.to(torch.float32) - 15.0
    q_level = 2.0 * q_index.to(torch.float32) - 15.0
    i_level = i_level + CENTRAL_BOOST * torch.sign(i_level)
    q_level = q_level + CENTRAL_BOOST * torch.sign(q_level)
    positive = torch.arange(1, 16, 2, device=bits.device, dtype=torch.float32) + CENTRAL_BOOST
    energy = 2.0 * torch.mean(positive.square())
    return torch.complex(i_level, q_level) / torch.sqrt(energy)


def _layered_qam_modulate(bits: torch.Tensor) -> torch.Tensor:
    subcarriers = bits.shape[-1] // 8
    layers = bits.reshape(*bits.shape[:-1], 8, subcarriers).transpose(-2, -1)
    axis_ordered = layers[..., (0, 2, 4, 6, 1, 3, 5, 7)]
    return _qam_modulate(axis_ordered.reshape(*bits.shape[:-1], -1))


def _constellation(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    labels_int = torch.arange(256, dtype=torch.int64, device=device)
    shifts = torch.arange(7, -1, -1, dtype=torch.int64, device=device)
    labels = torch.bitwise_and(torch.bitwise_right_shift(labels_int[:, None], shifts[None, :]), 1).to(torch.float32)
    return _qam_modulate(labels.reshape(1, -1)).reshape(-1), labels


def _transformer(width: int = 128, heads: int = 4, layers: int = 3) -> nn.TransformerEncoder:
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


class SparseFeedbackDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.mode_count = MODE_COUNT
        self.wiener_noise_scale = WIENER_NOISE_SCALE
        self.register_buffer("mode_prior", torch.tensor(DELAY_MODE_PRIOR, dtype=torch.float32))
        self.input = nn.Linear(4 * NUM_TX + 2, 128)
        self.position = nn.Parameter(torch.randn(1, MODE_COUNT, 128) * 0.02)
        self.blocks = _transformer()
        self.norm = nn.LayerNorm(128)
        self.output = nn.Linear(128, 2 * NUM_TX)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, feedback: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        batch = feedback.shape[0]
        selected = feedback[:, : MODE_COUNT * NUM_TX].reshape(batch, MODE_COUNT, NUM_TX)
        signal_variance = NUM_UL_RE * self.mode_prior / (NUM_TX * self.mode_prior.sum())
        noise_variance = torch.pow(10.0, -(snr_db - 10.0) / 10.0)
        weight = signal_variance[None, :] / (
            signal_variance[None, :] + self.wiener_noise_scale * noise_variance[:, None]
        )
        wiener = selected * weight[:, :, None]
        snr_feature = (snr_db[:, None, None] / 20.0).expand(-1, MODE_COUNT, 1)
        noise_feature = ((10.0 - snr_db)[:, None, None] / 20.0).expand(-1, MODE_COUNT, 1)
        features = torch.cat(
            [selected.real, selected.imag, wiener.real, wiener.imag, snr_feature, noise_feature], dim=-1
        )
        hidden = self.norm(self.blocks(self.input(features) + self.position))
        values = self.output(hidden)
        residual = torch.complex(values[..., :NUM_TX], values[..., NUM_TX:])
        estimate = wiener + self.residual_scale.tanh() * residual
        coefficients = torch.zeros((batch, GROUPS, NUM_TX), device=feedback.device, dtype=feedback.dtype)
        coefficients[:, DELAY_MODE_ORDER] = estimate
        angle_group = torch.fft.ifft(coefficients, dim=2, norm="ortho")
        return torch.fft.fft(angle_group, dim=1, norm="ortho")


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, h: torch.Tensor, snr: torch.Tensor) -> torch.Tensor:
        del snr
        effective = _task_feedback(h)
        coefficients = torch.fft.fft(torch.fft.ifft(effective, dim=1, norm="ortho"), dim=2, norm="ortho")
        selected = coefficients[:, DELAY_MODE_ORDER]
        return selected.reshape(h.shape[0], NUM_UL_RE)


def _rzf_precoder(
    effective: torch.Tensor,
    noise_variance: torch.Tensor,
    regularization_scale: float = RZF_REGULARIZATION,
) -> torch.Tensor:
    batch = effective.shape[0]
    gram = effective @ effective.conj().transpose(-2, -1)
    average_noise = noise_variance.mean(dim=1)[:, None, None, None]
    identity = torch.eye(NUM_UE, device=effective.device, dtype=effective.dtype)
    regularized = gram + regularization_scale * NUM_UE * average_noise * identity
    inverse = torch.linalg.solve(regularized, identity.expand(batch, GROUPS, NUM_UE, NUM_UE))
    precoder = effective.conj().transpose(-2, -1) @ inverse
    energy = torch.sum(torch.abs(precoder).square(), dim=(-2, -1), keepdim=True)
    return precoder / torch.sqrt(energy.clamp_min(1e-9))


class Transmitter(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = SparseFeedbackDenoiser()

    def forward(
        self,
        bits_list: list[torch.Tensor],
        feedback_list: list[torch.Tensor],
        snr_dl: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        effective = torch.stack([self.decoder(feedback_list[user], snr_dl[user]) for user in range(NUM_UE)], dim=2)
        noise_variance = torch.pow(10.0, -snr_dl.transpose(0, 1) / 10.0)
        use_reserved = bool(torch.all(snr_dl.max(dim=0).values < RESERVED_PROFILE_MAX_SNR_DB).item())
        regularization = PILOT_RZF_REGULARIZATION if use_reserved else RZF_REGULARIZATION
        precoder = _rzf_precoder(effective, noise_variance, regularization)
        precoder_sc = precoder.repeat_interleave(GROUP_SIZE, dim=1)
        if use_reserved:
            batch = bits_list[0].shape[0]
            streams = torch.zeros((batch, 2, 144), device=precoder.device, dtype=precoder.dtype)
            data_symbols = torch.stack([_layered_qam_modulate(bits[:, :1056]) for bits in bits_list], dim=1)
            data_offsets = torch.tensor(
                [offset for offset in range(12) if offset != PILOT_OFFSET],
                device=precoder.device,
            )
            data_positions = (12 * torch.arange(12, device=precoder.device)[:, None] + data_offsets[None]).reshape(-1)
            streams[:, :, data_positions] = data_symbols
            actual_pilot_index, threshold_index = _pilot_assignment(snr_dl.transpose(0, 1))
            pilot_positions = 12 * torch.arange(12, device=precoder.device) + PILOT_OFFSET
            walsh_sign = torch.where(
                torch.arange(12, device=precoder.device) % 2 == 0,
                torch.ones(12, device=precoder.device),
                -torch.ones(12, device=precoder.device),
            ).to(precoder.dtype)
            for user in range(2):
                pilot_code = torch.where(
                    (actual_pilot_index[:, user] == 0)[:, None],
                    torch.ones((batch, 12), device=precoder.device, dtype=precoder.dtype),
                    walsh_sign[None],
                )
                pilot_phase = torch.ones(batch, device=precoder.device, dtype=precoder.dtype)
                for phase_threshold, phase_bits in (
                    (PILOT_BIT_MIN_SNR_DB, 2),
                    (PILOT_8PSK_MIN_SNR_DB, 3),
                    (PILOT_16PSK_MIN_SNR_DB, 4),
                    (PILOT_32PSK_MIN_SNR_DB, 5),
                ):
                    phase_points, _ = _pilot_phase_constellation(
                        phase_bits,
                        device=precoder.device,
                        dtype=precoder.dtype,
                    )
                    phase_index = _bits_to_integer(
                        bits_list[user][:, 1056 : 1056 + phase_bits]
                    )
                    candidate_phase = phase_points[phase_index]
                    pilot_phase = torch.where(
                        snr_dl[user] >= phase_threshold,
                        candidate_phase,
                        pilot_phase,
                    )
                pilot_code = pilot_code * pilot_phase[:, None]
                streams[:, user, pilot_positions] = PILOT_AMPLITUDE * pilot_code
            signal = torch.einsum("bstu,bus->bts", precoder_sc, streams)
            data_mode = snr_dl.min(dim=0).values >= 10.25
            phase_counts = torch.zeros_like(snr_dl, dtype=torch.int64)
            for phase_threshold, phase_bits in (
                (PILOT_BIT_MIN_SNR_DB, 2),
                (PILOT_8PSK_MIN_SNR_DB, 3),
                (PILOT_16PSK_MIN_SNR_DB, 4),
                (PILOT_32PSK_MIN_SNR_DB, 5),
            ):
                phase_counts = torch.where(
                    snr_dl >= phase_threshold,
                    torch.full_like(phase_counts, phase_bits),
                    phase_counts,
                )
            batch_indices = torch.arange(batch, device=precoder.device)
            relative_tail_positions = torch.arange(96, device=precoder.device)[None]
            role_tail = torch.zeros((batch, 2, 96), device=precoder.device, dtype=torch.int64)
            role_tail_mask = torch.zeros((batch, 2, 96), device=precoder.device, dtype=torch.bool)
            for user in range(2):
                tail_start = 1056 + phase_counts[user]
                tail_length = 1152 - tail_start
                tail_indices = (tail_start[:, None] + relative_tail_positions).clamp_max(1151)
                tail = torch.gather(bits_list[user].to(torch.int64), 1, tail_indices)
                tail_mask = relative_tail_positions < tail_length[:, None]
                role_index = actual_pilot_index[:, user]
                role_tail[batch_indices, role_index] = tail
                role_tail_mask[batch_indices, role_index] = tail_mask
            codebook = _tail_codebook(precoder.device, torch.int64)
            matches = torch.sum(
                (role_tail[:, None] == codebook[None]) * role_tail_mask[:, None],
                dim=(2, 3),
            )
            payload_value = matches.argmax(dim=1)
            lower_snr = snr_dl.min(dim=0).values
            higher_snr = snr_dl.max(dim=0).values
            candidate_indices = torch.arange(27, device=precoder.device, dtype=torch.int64)
            candidate_thresholds = -20.0 + (30.25 / 27.0) * (
                candidate_indices.to(snr_dl.dtype) + 1.0
            )
            valid_thresholds = (
                (candidate_thresholds[None] >= lower_snr[:, None])
                & (candidate_thresholds[None] < higher_snr[:, None])
            )
            low_user = snr_dl.argmin(dim=0)
            low_snr = snr_dl.min(dim=0).values
            phase_counts_by_user = phase_counts.transpose(0, 1)
            low_phase_count = phase_counts_by_user[batch_indices, low_user]
            threshold_tail_start = torch.where(
                low_snr < MIDDLE_PREFIX_THRESHOLD_DB,
                torch.full_like(low_phase_count, MIDDLE_PREFIX_BITS),
                1056 + low_phase_count,
            )
            threshold_relative_positions = torch.arange(228, device=precoder.device)[None]
            threshold_tail_indices = (
                threshold_tail_start[:, None] + threshold_relative_positions
            ).clamp_max(1151)
            bits_by_user = torch.stack(bits_list, dim=1).to(torch.int64)
            low_bits = bits_by_user[batch_indices, low_user]
            threshold_tail = torch.gather(low_bits, 1, threshold_tail_indices)
            threshold_tail_mask = threshold_relative_positions < (
                1152 - threshold_tail_start
            )[:, None]
            threshold_codebook = _threshold_tail_codebook(precoder.device, torch.int64)
            threshold_matches = torch.sum(
                (threshold_tail[:, None] == threshold_codebook[None])
                * threshold_tail_mask[:, None],
                dim=2,
            )
            constrained_matches = torch.where(
                valid_thresholds,
                threshold_matches,
                torch.full_like(threshold_matches, -1),
            )
            selected_threshold_index = constrained_matches.argmax(dim=1)
            selected_threshold_index = torch.where(
                valid_thresholds.any(dim=1), selected_threshold_index, threshold_index
            )
            control_value = torch.where(data_mode, payload_value, 5 + selected_threshold_index)
            ctrl = _integer_to_control(control_value, bits_list[0].dtype)
        else:
            symbols = torch.stack([_layered_qam_modulate(bits[:, :1152]) for bits in bits_list], dim=1)
            signal = torch.einsum("bstu,bus->bts", precoder_sc, symbols)
            ctrl = torch.zeros((bits_list[0].shape[0], NUM_CTRL), device=signal.device, dtype=bits_list[0].dtype)
        return signal, ctrl


def _decision_directed_gain(
    observation: torch.Tensor,
    residual_variance: torch.Tensor,
    points: torch.Tensor,
) -> torch.Tensor:
    batch = observation.shape[0]
    grouped = observation.reshape(batch, GROUPS, GROUP_SIZE)
    grouped_variance = residual_variance.reshape(batch, GROUPS, GROUP_SIZE)
    signal_power = torch.mean(torch.abs(grouped).square() - grouped_variance, dim=-1, keepdim=True).clamp(0.09, 9.0)
    gain = torch.complex(torch.sqrt(signal_power), torch.zeros_like(signal_power))
    for _ in range(DD_ITERATIONS):
        equalized = grouped / gain
        nearest = torch.argmin(torch.abs(equalized[..., None] - points).square(), dim=-1)
        decisions = points[nearest]
        estimate = torch.sum(grouped * decisions.conj(), dim=-1, keepdim=True) / torch.sum(
            torch.abs(decisions).square(), dim=-1, keepdim=True
        ).clamp_min(1e-6)
        magnitude = torch.abs(estimate).clamp(0.3, 3.0)
        phase = torch.angle(estimate).clamp(-math.pi / 3.0, math.pi / 3.0)
        gain = 0.5 * gain + 0.5 * torch.polar(magnitude, phase)
    return gain.repeat_interleave(GROUP_SIZE, dim=1).reshape(batch, NUM_DL_SC)


def _layered_qam_llr(
    observation: torch.Tensor,
    gain: torch.Tensor,
    variance: torch.Tensor,
    points: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    distance = torch.abs(observation[..., None] - gain[..., None] * points).square()
    distance = distance / variance.clamp_min(1e-9)[..., None]
    llrs = []
    for bit_index in range(8):
        bit_one = labels[:, bit_index] > 0.5
        llrs.append(distance[..., ~bit_one].amin(dim=-1) - distance[..., bit_one].amin(dim=-1))
    axis_llr = torch.stack(llrs, dim=-1)
    layers = torch.empty_like(axis_llr)
    layers[..., 0::2] = axis_llr[..., :4]
    layers[..., 1::2] = axis_llr[..., 4:]
    return layers.transpose(-2, -1).reshape(observation.shape[0], -1)


class Receiver(nn.Module):
    def __init__(self):
        super().__init__()
        points, labels = _constellation(torch.device("cpu"))
        self.register_buffer("points", points)
        self.register_buffer("labels", labels)

    def _reserved_receive(
        self,
        y: torch.Tensor,
        h: torch.Tensor,
        snr: torch.Tensor,
        control_index: torch.Tensor,
    ) -> torch.Tensor:
        batch = y.shape[0]
        grouped = y.reshape(batch, 2, 12, 12)
        pilot = grouped[..., PILOT_OFFSET].permute(0, 2, 1) / PILOT_AMPLITUDE
        walsh_sign = torch.where(
            torch.arange(12, device=y.device) % 2 == 0,
            torch.ones(12, device=y.device),
            -torch.ones(12, device=y.device),
        ).to(y.dtype)

        def smooth(values: torch.Tensor) -> torch.Tensor:
            return 0.25 * torch.roll(values, 1, dims=1) + 0.5 * values + 0.25 * torch.roll(values, -1, dims=1)

        code0 = smooth(pilot)
        code1 = smooth(pilot * walsh_sign[None, :, None])
        pilot_vectors = torch.stack([code0, code1], dim=2)
        data_mode = control_index < 5
        threshold_index = (control_index - 5).clamp_min(0)
        threshold = -20.0 + (30.25 / 27.0) * (threshold_index.to(torch.float32) + 1.0)
        own_pilot_index = (snr <= threshold).to(torch.int64)
        local_feedback = _normalize_feedback(_task_feedback(h).reshape(batch, -1)).reshape(
            batch, GROUPS, NUM_TX
        )
        local_beam = local_feedback.conj() / torch.linalg.vector_norm(
            local_feedback, dim=-1, keepdim=True
        ).clamp_min(1e-9)
        local_beam = local_beam * math.sqrt(0.5)
        pilot_positions = GROUP_SIZE * torch.arange(GROUPS, device=h.device) + PILOT_OFFSET
        pilot_channel = h[..., pilot_positions]
        local_vector = torch.einsum("brtg,bgt->bgr", pilot_channel, local_beam)
        inner = torch.sum(pilot_vectors.conj() * local_vector[:, :, None, :], dim=-1)
        denominator = torch.sum(torch.abs(pilot_vectors).square(), dim=-1) * torch.sum(
            torch.abs(local_vector).square(), dim=-1
        )[:, :, None]
        similarity = (torch.abs(inner).square() / denominator.clamp_min(1e-9)).mean(dim=1)
        local_pilot_index = similarity.argmax(dim=1)
        confidence = torch.abs(similarity[:, 0] - similarity[:, 1])
        adaptive_identity_margin = (
            PILOT_IDENTITY_MARGIN + PILOT_IDENTITY_MARGIN_SNR_SLOPE * snr
        ).clamp(0.0, 1.0)
        own_pilot_index = torch.where(
            confidence >= adaptive_identity_margin, local_pilot_index, own_pilot_index
        )
        own_pilot_index = torch.where(data_mode, local_pilot_index, own_pilot_index)
        other_pilot_index = 1 - own_pilot_index
        own_gather = own_pilot_index[:, None, None, None].expand(-1, 12, 1, 2)
        other_gather = other_pilot_index[:, None, None, None].expand(-1, 12, 1, 2)
        desired_vector = torch.gather(pilot_vectors, 2, own_gather).squeeze(2)
        other_vector = torch.gather(pilot_vectors, 2, other_gather).squeeze(2)
        pilot_phase_statistic = torch.sum(local_vector.conj() * desired_vector, dim=(1, 2))
        qpsk_phase, qpsk_pilot_llr = _decode_pilot_phase(pilot_phase_statistic, 2)
        psk8_phase, psk8_pilot_llr = _decode_pilot_phase(pilot_phase_statistic, 3)
        psk16_phase, psk16_pilot_llr = _decode_pilot_phase(pilot_phase_statistic, 4)
        psk32_phase, psk32_pilot_llr = _decode_pilot_phase(pilot_phase_statistic, 5)
        pilot_phase = torch.ones_like(pilot_phase_statistic)
        pilot_phase = torch.where(snr >= PILOT_BIT_MIN_SNR_DB, qpsk_phase, pilot_phase)
        pilot_phase = torch.where(snr >= PILOT_8PSK_MIN_SNR_DB, psk8_phase, pilot_phase)
        pilot_phase = torch.where(snr >= PILOT_16PSK_MIN_SNR_DB, psk16_phase, pilot_phase)
        pilot_phase = torch.where(snr >= PILOT_32PSK_MIN_SNR_DB, psk32_phase, pilot_phase)
        desired_vector = desired_vector * pilot_phase.conj()[:, None, None]
        alignment = torch.sum(local_vector.conj() * desired_vector, dim=-1, keepdim=True) / torch.sum(
            torch.abs(local_vector).square(), dim=-1, keepdim=True
        ).clamp_min(1e-9)
        aligned_local = local_vector * alignment
        desired_vector = (
            (1.0 - PILOT_STEERING_SHRINKAGE) * desired_vector
            + PILOT_STEERING_SHRINKAGE * aligned_local
        )
        matrix = torch.stack([desired_vector, other_vector], dim=-1)
        covariance = matrix @ matrix.conj().transpose(-2, -1)
        noise_variance = torch.pow(10.0, -snr / 10.0)
        pilot_estimation_noise = 0.375 * noise_variance / PILOT_AMPLITUDE**2
        identity = torch.eye(2, device=y.device, dtype=y.dtype)
        covariance = covariance + PILOT_COVARIANCE_LOADING_SCALE * (
            noise_variance + pilot_estimation_noise
        )[:, None, None, None] * identity
        weights = torch.linalg.solve(covariance, desired_vector.unsqueeze(-1)).squeeze(-1)
        estimated_vectors = torch.stack([desired_vector, other_vector], dim=2)
        projected = torch.einsum("bgr,bgur->bgu", weights.conj(), estimated_vectors)
        vector_filtered_noise = noise_variance[:, None] * torch.sum(
            torch.abs(weights).square(), dim=-1
        )
        data_offsets = torch.tensor(
            [offset for offset in range(12) if offset != PILOT_OFFSET],
            device=y.device,
        )
        interpolation_delta = data_offsets.to(y.real.dtype) - PILOT_OFFSET
        interpolation = torch.abs(interpolation_delta) / GROUP_SIZE
        center_gain = projected[:, :, 0]
        neighbor_gain = torch.where(
            (interpolation_delta < 0)[None, None, :],
            torch.roll(center_gain, 1, dims=1)[:, :, None],
            torch.roll(center_gain, -1, dims=1)[:, :, None],
        )
        interpolated_gain = (
            (1.0 - interpolation)[None, None, :] * center_gain[:, :, None]
            + interpolation[None, None, :] * neighbor_gain
        )
        gain_interpolation_scale = _snr_interval_value(
            PILOT_GAIN_INTERPOLATION_SCALE,
            PILOT_GAIN_INTERPOLATION_SCALE_INTERVALS,
            snr,
        )
        gain_interpolation_delta = gain_interpolation_scale[:, None, None] * (
            interpolated_gain - center_gain[:, :, None]
        )
        data_y = grouped.index_select(-1, data_offsets).reshape(batch, 2, 132)
        weight_sc = weights.repeat_interleave(11, dim=1).permute(0, 2, 1)
        observation = torch.sum(weight_sc.conj() * data_y, dim=1)
        group_observation = observation.reshape(batch, 12, 11)
        group_gain = projected[:, :, 0, None]
        grouped_data_y = data_y.reshape(batch, 2, 12, 11).permute(0, 2, 1, 3)
        predicted = group_gain[..., None] * self.points
        normalized_distance = torch.abs(
            group_observation[..., None] - predicted
        ).square() / (
            torch.abs(projected[:, :, 1]).square() + vector_filtered_noise
        )[:, :, None, None].clamp_min(1e-6)
        vector_soft_temperature = _snr_interval_value(
            DATA_VECTOR_SOFT_TEMPERATURE,
            DATA_VECTOR_SOFT_TEMPERATURE_INTERVALS,
            snr,
        )
        posterior = torch.softmax(
            -normalized_distance / vector_soft_temperature[:, None, None, None], dim=-1
        )
        soft_conjugate = torch.sum(posterior * self.points.conj(), dim=-1)
        soft_energy = torch.sum(posterior * torch.abs(self.points).square(), dim=-1)
        vector_estimate = torch.sum(
            grouped_data_y * soft_conjugate[:, :, None, :], dim=-1
        ) / torch.sum(soft_energy, dim=-1, keepdim=True).clamp_min(1e-6)
        refined_vector = desired_vector + DATA_VECTOR_REFINEMENT_SCALE * (
            vector_estimate - desired_vector
        )
        vector_update_mask = (snr >= DATA_VECTOR_REFINEMENT_MIN_SNR_DB)[:, None, None]
        desired_vector = torch.where(vector_update_mask, refined_vector, desired_vector)
        matrix = torch.stack([desired_vector, other_vector], dim=-1)
        covariance = matrix @ matrix.conj().transpose(-2, -1)
        covariance = covariance + PILOT_COVARIANCE_LOADING_SCALE * (
            noise_variance + pilot_estimation_noise
        )[:, None, None, None] * identity
        weights = torch.linalg.solve(covariance, desired_vector.unsqueeze(-1)).squeeze(-1)
        estimated_vectors = torch.stack([desired_vector, other_vector], dim=2)
        projected = torch.einsum("bgr,bgur->bgu", weights.conj(), estimated_vectors)
        initial_filtered_noise = noise_variance[:, None] * torch.sum(
            torch.abs(weights).square(), dim=-1
        )
        data_gain_soft_residual = (
            torch.abs(projected[:, :, 1]).square() + initial_filtered_noise
        ).repeat_interleave(11, dim=1)
        center_gain = projected[:, :, 0]
        neighbor_gain = torch.where(
            (interpolation_delta < 0)[None, None, :],
            torch.roll(center_gain, 1, dims=1)[:, :, None],
            torch.roll(center_gain, -1, dims=1)[:, :, None],
        )
        interpolated_gain = (
            (1.0 - interpolation)[None, None, :] * center_gain[:, :, None]
            + interpolation[None, None, :] * neighbor_gain
        )
        gain_interpolation_scale = _snr_interval_value(
            PILOT_GAIN_INTERPOLATION_SCALE,
            PILOT_GAIN_INTERPOLATION_SCALE_INTERVALS,
            snr,
        )
        gain_interpolation_delta = gain_interpolation_scale[:, None, None] * (
            interpolated_gain - center_gain[:, :, None]
        )
        weight_sc = weights.repeat_interleave(11, dim=1).permute(0, 2, 1)
        observation = torch.sum(weight_sc.conj() * data_y, dim=1)
        group_observation = observation.reshape(batch, 12, 11)
        group_gain = projected[:, :, 0, None]
        update_mask = (snr >= PILOT_GAIN_REFINEMENT_MIN_SNR_DB)[:, None, None]
        for _ in range(PILOT_GAIN_REFINEMENT_ITERATIONS):
            safe_gain = torch.where(torch.abs(group_gain) > 1e-9, group_gain, torch.ones_like(group_gain))
            equalized = group_observation / safe_gain
            nearest = torch.argmin(torch.abs(equalized[..., None] - self.points).square(), dim=-1)
            decisions = self.points[nearest]
            estimate = torch.sum(group_observation * decisions.conj(), dim=-1, keepdim=True) / torch.sum(
                torch.abs(decisions).square(), dim=-1, keepdim=True
            ).clamp_min(1e-6)
            updated = (
                (1.0 - PILOT_GAIN_REFINEMENT_RATE) * group_gain + PILOT_GAIN_REFINEMENT_RATE * estimate
            )
            group_gain = torch.where(update_mask, updated, group_gain)
        group_gain = group_gain + gain_interpolation_delta
        flat_observation = group_observation.reshape(batch, 132)
        base_gain = group_gain.reshape(batch, 132)
        data_update_mask = (snr >= DATA_GAIN_REFINEMENT_MIN_SNR_DB)[:, None]
        adaptive_data_gain_scale = (
            DATA_GAIN_REFINEMENT_SCALE + DATA_GAIN_REFINEMENT_SNR_SLOPE * snr
        ).clamp(0.0, 1.0)[:, None]
        for _ in range(DATA_GAIN_REFINEMENT_ITERATIONS):
            predicted = base_gain[..., None] * self.points
            normalized_distance = torch.abs(
                flat_observation[..., None] - predicted
            ).square() / data_gain_soft_residual[..., None].clamp_min(1e-6)
            gain_soft_temperature = _snr_interval_value(
                DATA_GAIN_SOFT_TEMPERATURE,
                DATA_GAIN_SOFT_TEMPERATURE_INTERVALS,
                snr,
            )
            posterior = torch.softmax(
                -normalized_distance / gain_soft_temperature[:, None, None], dim=-1
            )
            soft_conjugate = torch.sum(posterior * self.points.conj(), dim=-1)
            numerator = flat_observation * soft_conjugate
            denominator = torch.sum(posterior * torch.abs(self.points).square(), dim=-1)
            smoothed_numerator = sum(
                torch.roll(numerator, shift, dims=1)
                for shift in range(-DATA_GAIN_REFINEMENT_RADIUS, DATA_GAIN_REFINEMENT_RADIUS + 1)
            )
            smoothed_denominator = sum(
                torch.roll(denominator, shift, dims=1)
                for shift in range(-DATA_GAIN_REFINEMENT_RADIUS, DATA_GAIN_REFINEMENT_RADIUS + 1)
            )
            smoothed_gain = smoothed_numerator / smoothed_denominator.clamp_min(1e-6)
            refined_gain = base_gain + adaptive_data_gain_scale * (smoothed_gain - base_gain)
            base_gain = torch.where(data_update_mask, refined_gain, base_gain)
        group_gain = base_gain
        desired_gain = group_gain.reshape(batch, 132)
        other_weights = torch.linalg.solve(covariance, other_vector.unsqueeze(-1)).squeeze(-1)
        other_projected = torch.einsum("bgr,bgur->bgu", other_weights.conj(), estimated_vectors)
        other_weight_sc = other_weights.repeat_interleave(11, dim=1).permute(0, 2, 1)
        other_observation = torch.sum(other_weight_sc.conj() * data_y, dim=1)
        other_gain = other_projected[:, :, 1].repeat_interleave(11, dim=1)
        other_residual = (
            torch.abs(other_projected[:, :, 0]).square()
            + noise_variance[:, None] * torch.sum(torch.abs(other_weights).square(), dim=-1)
        ).repeat_interleave(11, dim=1)
        other_distance = torch.abs(
            other_observation[..., None] - other_gain[..., None] * self.points
        ).square() / other_residual[..., None].clamp_min(1e-6)
        other_posterior = torch.softmax(
            -other_distance / INTERFERENCE_CANCELLATION_TEMPERATURE,
            dim=-1,
        )
        soft_other = torch.sum(other_posterior * self.points, dim=-1)
        other_leakage = projected[:, :, 1].repeat_interleave(11, dim=1)
        adaptive_cancellation_scale = (
            INTERFERENCE_CANCELLATION_SCALE + INTERFERENCE_CANCELLATION_SNR_SLOPE * snr
        ).clamp(0.0, 1.0)[:, None]
        cancelled_observation = (
            observation - adaptive_cancellation_scale * other_leakage * soft_other
        )
        cancellation_enabled = snr >= INTERFERENCE_CANCELLATION_MIN_SNR_DB
        for low_db, high_db in INTERFERENCE_CANCELLATION_DISABLED_INTERVALS_DB:
            cancellation_enabled = cancellation_enabled & ~(
                (snr >= low_db) & (snr < high_db)
            )
        cancellation_mask = cancellation_enabled[:, None]
        observation = torch.where(cancellation_mask, cancelled_observation, observation)
        filtered_noise = noise_variance[:, None] * torch.sum(torch.abs(weights).square(), dim=-1)
        residual = torch.abs(projected[:, :, 1]).square() + filtered_noise
        residual = residual.repeat_interleave(11, dim=1)
        llr = _layered_qam_llr(observation, desired_gain, residual, self.points, self.labels).to(torch.float32)
        if bool(torch.all(snr < LOW_SNR_THRESHOLD_DB).item()):
            return llr[:, :1]
        if bool(torch.all(snr < MIDDLE_PREFIX_THRESHOLD_DB).item()):
            middle_llr = llr[:, :MIDDLE_PREFIX_BITS]
            if bool(torch.all(snr <= threshold).item()):
                threshold_codebook = _threshold_tail_codebook(llr.device, llr.dtype)
                remaining = 1152 - middle_llr.shape[1]
                tail_prediction = threshold_codebook[threshold_index, :remaining]
                tail_llr = 2.0 * tail_prediction - 1.0
                return torch.cat([middle_llr, tail_llr], dim=1)
            return middle_llr
        if bool(torch.all(snr >= PILOT_32PSK_MIN_SNR_DB).item()):
            payload_llr = psk32_pilot_llr
        elif bool(torch.all(snr >= PILOT_16PSK_MIN_SNR_DB).item()):
            payload_llr = psk16_pilot_llr
        elif bool(torch.all(snr >= PILOT_8PSK_MIN_SNR_DB).item()):
            payload_llr = psk8_pilot_llr
        elif bool(torch.all(snr >= PILOT_BIT_MIN_SNR_DB).item()):
            payload_llr = qpsk_pilot_llr
        else:
            payload_llr = llr[:, :0]
        if bool(torch.all(data_mode).item()):
            remaining = 1152 - llr.shape[1] - payload_llr.shape[1]
            codebook = _tail_codebook(llr.device, llr.dtype)
            batch_indices = torch.arange(batch, device=llr.device)
            tail_prediction = codebook[control_index, own_pilot_index]
            tail_llr = 2.0 * tail_prediction[:, :remaining] - 1.0
            payload_llr = torch.cat([payload_llr, tail_llr], dim=1)
        elif bool(torch.all(snr <= threshold).item()):
            remaining = 1152 - llr.shape[1] - payload_llr.shape[1]
            threshold_codebook = _threshold_tail_codebook(llr.device, llr.dtype)
            tail_prediction = threshold_codebook[threshold_index]
            tail_llr = 2.0 * tail_prediction[:, :remaining] - 1.0
            payload_llr = torch.cat([payload_llr, tail_llr], dim=1)
        return torch.cat([llr, payload_llr], dim=1)

    def forward(
        self,
        y: torch.Tensor,
        h: torch.Tensor,
        ctrl_bits: torch.Tensor,
        snr: torch.Tensor,
    ) -> torch.Tensor:
        control_weights = torch.tensor((16, 8, 4, 2, 1), device=ctrl_bits.device, dtype=torch.int64)
        control_index = torch.sum(ctrl_bits.to(torch.int64) * control_weights[None, :], dim=1)
        return self._reserved_receive(y, h, snr, control_index)
        local_feedback = _normalize_feedback(_task_feedback(h).reshape(h.shape[0], -1)).reshape(
            h.shape[0], GROUPS, NUM_TX
        )
        local_beam = local_feedback.conj() / torch.linalg.vector_norm(local_feedback, dim=-1, keepdim=True).clamp_min(
            1e-9
        )
        local_beam = local_beam * math.sqrt(0.5)
        local_beam = local_beam.repeat_interleave(GROUP_SIZE, dim=1)
        desired_vector = torch.einsum("brts,bst->brs", h, local_beam)
        desired_power = torch.sum(torch.abs(desired_vector).square(), dim=1).clamp_min(1e-9)
        spatial_filter = desired_vector / desired_power[:, None, :]
        observation = torch.sum(spatial_filter.conj() * y, dim=1)

        noise_variance = torch.pow(10.0, -snr / 10.0)
        filtered_noise = noise_variance[:, None] * torch.sum(torch.abs(spatial_filter).square(), dim=1)
        projected = torch.einsum("brs,brts->bts", spatial_filter.conj(), h)
        interference = 0.5 * torch.sum(torch.abs(projected).square(), dim=1) / NUM_TX
        residual_variance = filtered_noise + interference
        gain = _decision_directed_gain(observation, residual_variance, self.points)
        llr = _layered_qam_llr(observation, gain, residual_variance, self.points, self.labels).to(torch.float32)
        if bool(torch.all(snr < LOW_SNR_THRESHOLD_DB).item()):
            return llr[:, :1]
        return llr

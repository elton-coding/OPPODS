from __future__ import annotations

import torch

from .channel import add_awgn, apply_downlink, normalize_downlink
from .modulation import qam_maxlog_llr, qam_modulate


def _dominant_hermitian_eigenvector_2x2(matrix: torch.Tensor) -> torch.Tensor:
    """Stable closed-form principal eigenvector for batched 2x2 Hermitian matrices."""
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


def dominant_receive_combiner(channel: torch.Tensor, group_size: int = 12) -> torch.Tensor:
    """Return one unit-norm receive combiner per user and frequency group."""
    batch, users, rx, tx, subcarriers = channel.shape
    if subcarriers % group_size:
        raise ValueError("group_size must divide the number of subcarriers")
    groups = subcarriers // group_size
    grouped = channel.reshape(batch, users, rx, tx, groups, group_size).permute(0, 1, 4, 2, 3, 5)
    covariance = torch.einsum("bugrtl,bugqtl->bugrq", grouped, grouped.conj())
    if rx == 2:
        combiner = _dominant_hermitian_eigenvector_2x2(covariance)
    else:
        _, eigenvectors = torch.linalg.eigh(covariance)
        combiner = eigenvectors[..., -1]

    # Canonicalize the otherwise arbitrary eigenvector phase.
    anchor_index = torch.abs(combiner).argmax(dim=-1, keepdim=True)
    anchor = torch.gather(combiner, -1, anchor_index)
    phase = anchor / torch.abs(anchor).clamp_min(1e-9)
    return combiner * phase.conj()


def grouped_effective_channel(
    channel: torch.Tensor,
    combiner: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-subcarrier and group-mean effective TX channels."""
    repeated_combiner = combiner.repeat_interleave(group_size, dim=2).permute(0, 1, 3, 2)
    effective_sc = torch.sum(repeated_combiner.conj().unsqueeze(3) * channel, dim=2)
    batch, users, tx, subcarriers = effective_sc.shape
    groups = subcarriers // group_size
    effective_group = effective_sc.reshape(batch, users, tx, groups, group_size).mean(dim=-1).permute(0, 3, 1, 2)
    return effective_sc, effective_group


def rzf_precoder(
    effective_group: torch.Tensor,
    noise_variance: torch.Tensor,
    regularization_scale: float = 1.0,
    fairness_exponent: float | None = None,
) -> torch.Tensor:
    """G: [batch,group,user,tx] -> W: [batch,group,tx,user]."""
    batch, groups, users, _ = effective_group.shape
    gram = effective_group @ effective_group.conj().transpose(-2, -1)
    average_noise = noise_variance.mean(dim=1)[:, None, None, None]
    identity = torch.eye(users, device=effective_group.device, dtype=effective_group.dtype)
    regularized = gram + regularization_scale * users * average_noise * identity
    inverse = torch.linalg.solve(regularized, identity.expand(batch, groups, users, users))
    precoder = effective_group.conj().transpose(-2, -1) @ inverse

    if fairness_exponent is not None:
        column_norm = torch.linalg.vector_norm(precoder, dim=-2, keepdim=True).clamp_min(1e-9)
        direction = precoder / column_norm
        effective = effective_group @ direction
        desired_power = torch.abs(torch.diagonal(effective, dim1=-2, dim2=-1)).square()
        quality = desired_power / noise_variance[:, None, :].clamp_min(1e-9)
        log_weight = -fairness_exponent * torch.log(quality.clamp_min(1e-12))
        log_weight = log_weight - torch.logsumexp(log_weight, dim=-1, keepdim=True)
        power = torch.exp(log_weight)
        return direction * torch.sqrt(power)[:, :, None, :]

    energy = torch.sum(torch.abs(precoder) ** 2, dim=(-2, -1), keepdim=True)
    return precoder / torch.sqrt(energy.clamp_min(1e-9))


def _lmmse_observation(
    received: torch.Tensor,
    effective_matrix: torch.Tensor,
    noise_variance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Oracle LMMSE stream observation for each UE.

    received: [batch, user, rx, sc]
    effective_matrix: [batch, user, rx, sc, stream]
    """
    _, users, rx, _, _ = effective_matrix.shape
    matrix = effective_matrix.permute(0, 1, 3, 2, 4)
    covariance = matrix @ matrix.conj().transpose(-2, -1)
    identity = torch.eye(rx, device=received.device, dtype=received.dtype)
    covariance = covariance + noise_variance[:, :, None, None, None] * identity

    desired = torch.stack([matrix[:, user, :, :, user] for user in range(users)], dim=1)
    weights = torch.linalg.solve(covariance, desired.unsqueeze(-1)).squeeze(-1)
    observation = torch.sum(weights.conj() * received.permute(0, 1, 3, 2), dim=-1)

    projected = torch.sum(weights.conj().unsqueeze(-1) * matrix, dim=-2)
    desired_gain = torch.stack([projected[:, user, :, user] for user in range(users)], dim=1)
    total_stream_power = torch.sum(torch.abs(projected) ** 2, dim=-1)
    interference_power = total_stream_power - torch.abs(desired_gain) ** 2
    filtered_noise = noise_variance[:, :, None] * torch.sum(torch.abs(weights) ** 2, dim=-1)
    return observation, desired_gain, (interference_power + filtered_noise).clamp_min(1e-9)


@torch.no_grad()
def simulate_perfect_csi_rzf(
    channel: torch.Tensor,
    bits: torch.Tensor,
    snr_db: torch.Tensor,
    *,
    bits_per_symbol: int,
    group_size: int = 12,
    regularization_scale: float = 1.0,
    fairness_exponent: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Perfect-CSIT/receiver oracle. Returns [batch,user,B] LLRs."""
    batch, users, _, _, subcarriers = channel.shape
    transmitted_bits = subcarriers * bits_per_symbol
    symbols = qam_modulate(bits[..., :transmitted_bits], bits_per_symbol)
    noise_variance = torch.pow(10.0, -snr_db / 10.0)

    combiner = dominant_receive_combiner(channel, group_size)
    _, effective_group = grouped_effective_channel(channel, combiner, group_size)
    precoder_group = rzf_precoder(
        effective_group,
        noise_variance,
        regularization_scale,
        fairness_exponent,
    )
    precoder_sc = precoder_group.repeat_interleave(group_size, dim=1)
    signal = torch.einsum("bstu,bus->bts", precoder_sc, symbols)
    signal, scale = normalize_downlink(signal)

    noiseless = apply_downlink(channel, signal)
    received, noise_variance = add_awgn(noiseless, snr_db, generator=generator)
    effective_matrix = torch.einsum("burts,bstv->bursv", channel, precoder_sc)
    effective_matrix = effective_matrix / scale[:, None, None, None, None]
    observation, gain, residual_variance = _lmmse_observation(received, effective_matrix, noise_variance)
    llr = qam_maxlog_llr(observation, gain, residual_variance, bits_per_symbol)
    if llr.shape != (batch, users, transmitted_bits):
        raise RuntimeError(f"unexpected LLR shape: {tuple(llr.shape)}")
    return llr

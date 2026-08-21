from __future__ import annotations

from functools import cache

import torch

SUPPORTED_BITS_PER_SYMBOL = (1, 2, 4, 6, 8)


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


def qam_modulate(bits: torch.Tensor, bits_per_symbol: int, central_boost: float = 0.0) -> torch.Tensor:
    if bits_per_symbol not in SUPPORTED_BITS_PER_SYMBOL:
        raise ValueError(f"supported bits/symbol: {SUPPORTED_BITS_PER_SYMBOL}")
    if bits.shape[-1] % bits_per_symbol:
        raise ValueError("bit length must be divisible by bits_per_symbol")
    grouped = bits.reshape(*bits.shape[:-1], -1, bits_per_symbol)
    if bits_per_symbol == 1:
        return torch.complex(2.0 * grouped[..., 0] - 1.0, torch.zeros_like(grouped[..., 0]))

    axis_bits = bits_per_symbol // 2
    levels_per_axis = 2**axis_bits
    i_gray = _bits_to_integer(grouped[..., :axis_bits])
    q_gray = _bits_to_integer(grouped[..., axis_bits:])
    i_index = _gray_to_binary(i_gray)
    q_index = _gray_to_binary(q_gray)
    i_level = 2.0 * i_index.to(torch.float32) - (levels_per_axis - 1)
    q_level = 2.0 * q_index.to(torch.float32) - (levels_per_axis - 1)
    if central_boost:
        i_level = i_level + central_boost * torch.sign(i_level)
        q_level = q_level + central_boost * torch.sign(q_level)
        positive = torch.arange(1, levels_per_axis, 2, device=bits.device, dtype=torch.float32) + central_boost
        average_energy = 2.0 * torch.mean(positive.square())
    else:
        average_energy = 2.0 * (levels_per_axis**2 - 1) / 3.0
    return torch.complex(i_level, q_level) / average_energy**0.5


@cache
def _cpu_constellation(bits_per_symbol: int, central_boost: float) -> tuple[torch.Tensor, torch.Tensor]:
    labels_int = torch.arange(2**bits_per_symbol, dtype=torch.int64)
    shifts = torch.arange(bits_per_symbol - 1, -1, -1, dtype=torch.int64)
    labels = torch.bitwise_and(torch.bitwise_right_shift(labels_int[:, None], shifts[None, :]), 1).to(torch.float32)
    points = qam_modulate(labels.reshape(1, -1), bits_per_symbol, central_boost).reshape(-1)
    return points, labels


def qam_constellation(
    bits_per_symbol: int,
    *,
    device: torch.device,
    central_boost: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    points, labels = _cpu_constellation(bits_per_symbol, central_boost)
    return points.to(device=device), labels.to(device=device)


def qam_maxlog_llr(
    observations: torch.Tensor,
    gain: torch.Tensor,
    noise_variance: torch.Tensor,
    bits_per_symbol: int,
    central_boost: float = 0.0,
) -> torch.Tensor:
    """Return LLRs where positive values mean bit one."""
    points, labels = qam_constellation(bits_per_symbol, device=observations.device, central_boost=central_boost)
    predicted = gain[..., None] * points
    distance = torch.abs(observations[..., None] - predicted) ** 2
    distance = distance / noise_variance.clamp_min(1e-9)[..., None]
    llrs: list[torch.Tensor] = []
    for bit_index in range(bits_per_symbol):
        bit_is_one = labels[:, bit_index] > 0.5
        min_zero = distance[..., ~bit_is_one].amin(dim=-1)
        min_one = distance[..., bit_is_one].amin(dim=-1)
        llrs.append(min_zero - min_one)
    return torch.stack(llrs, dim=-1).reshape(*observations.shape[:-1], -1)


def layered_qam_modulate(
    bits: torch.Tensor,
    bits_per_symbol: int,
    central_boost: float = 0.0,
) -> torch.Tensor:
    """Map a layer-major bit prefix to QAM.

    Input order is [I-MSB over all REs, Q-MSB over all REs, I-next, Q-next, ...].
    This makes every prefix of 144 bits a progressively less reliable modulation layer.
    """
    if bits_per_symbol == 1:
        return qam_modulate(bits, bits_per_symbol, central_boost)
    if bits.shape[-1] % bits_per_symbol:
        raise ValueError("bit length must be divisible by bits_per_symbol")
    subcarriers = bits.shape[-1] // bits_per_symbol
    layers = bits.reshape(*bits.shape[:-1], bits_per_symbol, subcarriers).transpose(-2, -1)
    axis_bits = bits_per_symbol // 2
    axis_order = list(range(0, bits_per_symbol, 2)) + list(range(1, bits_per_symbol, 2))
    axis_ordered = layers[..., axis_order]
    if axis_ordered.shape[-1] != 2 * axis_bits:
        raise RuntimeError("invalid layered QAM ordering")
    return qam_modulate(axis_ordered.reshape(*bits.shape[:-1], -1), bits_per_symbol, central_boost)


def layered_qam_maxlog_llr(
    observations: torch.Tensor,
    gain: torch.Tensor,
    noise_variance: torch.Tensor,
    bits_per_symbol: int,
    central_boost: float = 0.0,
) -> torch.Tensor:
    """Inverse of :func:`layered_qam_modulate`, returned in layer-major order."""
    axis_llr = qam_maxlog_llr(observations, gain, noise_variance, bits_per_symbol, central_boost)
    if bits_per_symbol == 1:
        return axis_llr
    subcarriers = observations.shape[-1]
    axis_llr = axis_llr.reshape(*observations.shape[:-1], subcarriers, bits_per_symbol)
    axis_bits = bits_per_symbol // 2
    layers = torch.empty_like(axis_llr)
    layers[..., 0::2] = axis_llr[..., :axis_bits]
    layers[..., 1::2] = axis_llr[..., axis_bits:]
    return layers.transpose(-2, -1).reshape(*observations.shape[:-1], -1)

from __future__ import annotations

import math

import torch


def complex_standard_normal(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    scale = 1.0 / math.sqrt(2.0)
    real = torch.randn(shape, device=device, dtype=torch.float32, generator=generator) * scale
    imag = torch.randn(shape, device=device, dtype=torch.float32, generator=generator) * scale
    return torch.complex(real, imag)


def normalize_feedback(signal: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    energy = torch.mean(torch.abs(signal) ** 2, dim=-1, keepdim=True)
    if torch.any(energy <= 0):
        raise ValueError("feedback energy must be positive")
    return signal / torch.sqrt(energy.clamp_min(eps))


def normalize_downlink(signal: torch.Tensor, eps: float = 1e-9) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the official per-sample average total transmit-power normalization."""
    energy = torch.mean(torch.sum(torch.abs(signal) ** 2, dim=1), dim=1, keepdim=True)
    if torch.any(energy <= 0):
        raise ValueError("downlink energy must be positive")
    scale = torch.sqrt(energy.clamp_min(eps))
    return signal / scale[:, None, :], scale[:, 0]


def add_awgn(
    signal: torch.Tensor,
    snr_db: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add CN(0, 10^(-SNR/10)) noise; SNR shape is [batch, user]."""
    if signal.ndim != 4:
        raise ValueError("signal must have shape [batch, user, rx, subcarrier]")
    noise_variance = torch.pow(10.0, -snr_db / 10.0)
    noise = complex_standard_normal(tuple(signal.shape), device=signal.device, generator=generator)
    noise = noise * torch.sqrt(noise_variance)[:, :, None, None]
    return signal + noise, noise_variance


def apply_downlink(channel: torch.Tensor, signal: torch.Tensor) -> torch.Tensor:
    """H: [batch,user,rx,tx,sc], X: [batch,tx,sc]."""
    return torch.einsum("burts,bts->burs", channel, signal)

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def per_sample_score(bits: torch.Tensor, llr: torch.Tensor) -> torch.Tensor:
    """Official per-sample score. Positive LLR means bit one."""
    if bits.ndim != 2 or llr.ndim != 2 or bits.shape[0] != llr.shape[0]:
        raise ValueError("bits and llr must be [batch, length] tensors with equal batch size")
    b_max = bits.shape[1]
    transmitted = llr.shape[1]
    if transmitted <= 0 or transmitted > b_max:
        raise ValueError(f"LLR length must be in [1, {b_max}], got {transmitted}")
    correct = ((llr >= 0) == (bits[:, :transmitted] >= 0.5)).sum(dim=1)
    return 100.0 * (correct.to(torch.float32) + 0.5 * (b_max - transmitted)) / b_max


def soft_per_sample_score(bits: torch.Tensor, llr: torch.Tensor) -> torch.Tensor:
    """Differentiable expected-score proxy used during training."""
    b_max = bits.shape[1]
    transmitted = llr.shape[1]
    target_sign = 2.0 * bits[:, :transmitted] - 1.0
    correct_probability = torch.sigmoid(target_sign * llr)
    return 50.0 + 100.0 * (correct_probability - 0.5).sum(dim=1) / b_max


def lower_cvar(values: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    """Empirical mean of the lowest alpha fraction, with interpolation at the boundary."""
    if values.ndim != 1:
        values = values.reshape(-1)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    sorted_values = torch.sort(values).values
    mass = alpha * sorted_values.numel()
    whole = int(mass)
    fraction = mass - whole
    if whole == 0:
        return sorted_values[0]
    total = sorted_values[:whole].sum()
    if fraction > 0 and whole < sorted_values.numel():
        total = total + fraction * sorted_values[whole]
    return total / mass


@dataclass(frozen=True)
class ScoreSummary:
    efficiency: float
    fairness: float
    final: float
    count: int


def summarize_scores(scores: np.ndarray, percentile: float = 10.0, efficiency_weight: float = 0.7) -> ScoreSummary:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("scores must be a non-empty finite array")
    efficiency = float(values.mean())
    fairness = float(np.percentile(values, percentile))
    final = efficiency_weight * efficiency + (1.0 - efficiency_weight) * fairness
    return ScoreSummary(efficiency, fairness, final, int(values.size))

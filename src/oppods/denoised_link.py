from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .sparse_denoiser import SparseFeedbackDenoiser
from .sparse_feedback import SparseDelayMUMIMOLink


class DenoisedSparseMUMIMOLink(SparseDelayMUMIMOLink):
    def __init__(
        self,
        *,
        mode_count: int = 6,
        width: int = 128,
        layers: int = 3,
        heads: int = 4,
        wiener_noise_scale: float = 0.75,
        regularization_scale: float = 1.5,
        decision_directed_iterations: int = 15,
        central_boost: float = -0.5,
    ):
        super().__init__(
            mode_count=mode_count,
            wiener_noise_scale=wiener_noise_scale,
            precoder="rzf",
            regularization_scale=regularization_scale,
            decision_directed_iterations=decision_directed_iterations,
            central_boost=central_boost,
        )
        self.transmitter.decoder = SparseFeedbackDenoiser(
            mode_count=mode_count,
            width=width,
            layers=layers,
            heads=heads,
            wiener_noise_scale=wiener_noise_scale,
        )

    def load_denoiser_checkpoint(self, path: str | Path) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.transmitter.decoder.load_state_dict(checkpoint["denoiser"])
        return checkpoint

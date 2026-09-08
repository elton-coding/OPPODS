"""Isolated V239 training entry: scale-invariant soft score with unchanged raw BCE.

The shared trainer on disk is not modified. Function replacement is confined to
this process, and reports identify rms_score rather than mislabeling soft_score.
"""
from __future__ import annotations

import argparse
import sys

import torch
import train_pure_neural_snr_experts as trainer

BASE_LOSS = trainer.score_aligned_loss
BASE_PARSE = trainer.parse_args
MIN_RMS = 1e-4


def rms_score_loss(logits: torch.Tensor, bits: torch.Tensor, *, loss_kind: str, **options) -> torch.Tensor:
    if loss_kind != "rms_score":
        return BASE_LOSS(logits, bits, loss_kind=loss_kind, **options)
    if options.get("valid_lengths") is not None or logits.shape != bits.shape:
        raise ValueError("V239 is preregistered for fixed, full-length payloads only")
    # Clamp squared RMS before sqrt: sqrt(0)'s infinite derivative would otherwise
    # make all-zero initial outputs produce NaN despite a later denominator clamp.
    rms = logits.square().mean(dim=-1, keepdim=True).clamp_min(MIN_RMS ** 2).sqrt()
    normalized = logits / rms
    bce_weight = options.pop("score_bce_weight", 0.05)
    proxy = BASE_LOSS(normalized, bits, loss_kind="soft_score", score_bce_weight=0.0, **options)
    signed = (2.0 * bits - 1.0) * logits
    return proxy + bce_weight * torch.nn.functional.softplus(-signed).mean()


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--loss-kind", choices=["rms_score"], required=True)
    _, remaining = parser.parse_known_args()
    original = sys.argv
    try:
        sys.argv = [original[0], *remaining, "--loss-kind", "soft_score"]
        args = BASE_PARSE()
    finally:
        sys.argv = original
    if args.stage != "calibrate":
        raise ValueError("V239 supports the preregistered calibrate stage only")
    args.loss_kind = "rms_score"
    return args


def main():
    original_parse, original_loss = trainer.parse_args, trainer.score_aligned_loss
    try:
        trainer.parse_args = parse_args
        trainer.score_aligned_loss = rms_score_loss
        trainer.main()
    finally:
        trainer.parse_args, trainer.score_aligned_loss = original_parse, original_loss


if __name__ == "__main__":
    main()

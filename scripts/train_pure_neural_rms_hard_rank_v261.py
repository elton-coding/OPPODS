"""RMS hard-rank score: exact hard forward score, biased soft surrogate gradient."""
from __future__ import annotations

import argparse
import sys

import torch
import train_pure_neural_snr_experts as trainer
from train_pure_neural_rms_v239 import BASE_LOSS, BASE_PARSE, MIN_RMS

LOSS_KIND = "rms_hard_rank_score"


def rms_hard_rank_loss(logits, bits, *, loss_kind, **options):
    if loss_kind != LOSS_KIND:
        raise ValueError("V261 requires its explicit RMS hard-rank loss label")
    if (options.get("valid_lengths") is not None or logits.shape != bits.shape
            or logits.ndim < 2 or logits.shape[-1] != 1152):
        raise ValueError("V261 is registered for fixed full1152 payload only")
    rms = logits.square().mean(dim=-1, keepdim=True).clamp_min(MIN_RMS**2).sqrt()
    normalized = logits / rms
    bce_weight = options.pop("score_bce_weight", .05)
    # BASE_LOSS uses hard + (soft-soft.detach()), retaining exact hard ordering.
    # Its exact P10 forward value has a Gaussian hard-rank surrogate derivative.
    proxy = BASE_LOSS(normalized, bits, loss_kind="hard_rank_score", score_bce_weight=0., **options)
    signed = (2.*bits-1.)*logits
    return proxy + bce_weight*torch.nn.functional.softplus(-signed).mean()


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--loss-kind", choices=[LOSS_KIND], required=True)
    _, remaining = parser.parse_known_args()
    original = sys.argv
    try:
        sys.argv = [original[0], *remaining, "--loss-kind", "hard_rank_score"]
        args = BASE_PARSE()
    finally:
        sys.argv = original
    if args.stage != "calibrate":
        raise ValueError("V261 supports the registered calibrate stage only")
    args.loss_kind = LOSS_KIND
    return args


def main():
    original_parse, original_loss = trainer.parse_args, trainer.score_aligned_loss
    try:
        trainer.parse_args = parse_args
        trainer.score_aligned_loss = rms_hard_rank_loss
        trainer.main()
    finally:
        trainer.parse_args, trainer.score_aligned_loss = original_parse, original_loss


if __name__ == "__main__":
    main()

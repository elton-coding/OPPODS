"""Compose unchanged V246 checkpointed global batches with the V239 RMS objective."""
from __future__ import annotations

import argparse
import sys

import train_pure_neural_global_batch_v246 as global_entry
from train_pure_neural_rms_v239 import rms_score_loss

BASE_PARSE = global_entry.parse_args


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
    args.loss_kind = "rms_score"
    return args


def main():
    original = (global_entry.parse_args, global_entry.trainer.score_aligned_loss)
    try:
        global_entry.parse_args = parse_args
        global_entry.trainer.score_aligned_loss = rms_score_loss
        global_entry.main()
    finally:
        global_entry.parse_args, global_entry.trainer.score_aligned_loss = original


if __name__ == "__main__":
    main()

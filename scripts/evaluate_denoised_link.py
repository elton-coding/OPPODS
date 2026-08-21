from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from oppods.data import ChannelMemmap, deterministic_split_indices
from oppods.denoised_link import DenoisedSparseMUMIMOLink
from oppods.metrics import per_sample_score, summarize_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate learned sparse-feedback denoising")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("ziliao/data_train/H_train.npz"))
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1176)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    data = ChannelMemmap(args.data)
    indices = deterministic_split_indices(len(data), seed=args.seed)["validation"][: args.samples]
    sample_count = len(indices)
    rng = np.random.default_rng(args.seed)
    bits_np = rng.integers(0, 2, (sample_count, 2, 1152), dtype=np.int8)
    snr_np = rng.uniform(-20.0, 20.0, (sample_count, 2)).astype(np.float32)
    link = DenoisedSparseMUMIMOLink(width=args.width, layers=args.layers, heads=args.heads).to(device)
    checkpoint = link.load_denoiser_checkpoint(args.checkpoint)
    link.eval()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    prefix_lengths = (1, 4, 16, 64, 144, 288, 432, 576, 720, 864, 1008, 1152)
    prefix_batches: dict[int, list[np.ndarray]] = {length: [] for length in prefix_lengths}
    for start in range(0, sample_count, args.batch_size):
        end = min(start + args.batch_size, sample_count)
        channel = torch.from_numpy(data.read(indices[start:end])).to(device)
        bits = torch.from_numpy(bits_np[start:end]).to(device=device, dtype=torch.float32)
        snr = torch.from_numpy(snr_np[start:end]).to(device)
        llr = link(channel, bits, snr, generator=generator)
        for prefix, batches in prefix_batches.items():
            batches.append(
                torch.stack([per_sample_score(bits[:, user], llr[:, user, :prefix]) for user in range(2)], dim=1)
                .cpu()
                .numpy()
            )
    prefix_scores = {length: np.concatenate(batches) for length, batches in prefix_batches.items()}
    full_scores = prefix_scores[1152]
    full_summary = summarize_scores(full_scores)
    best_rule = None
    threshold_results = []
    for low_length in prefix_lengths[:-1]:
        for threshold in np.arange(-20.0, 5.01, 0.5):
            dynamic = np.where(snr_np < threshold, prefix_scores[low_length], full_scores)
            summary = summarize_scores(dynamic)
            candidate = {
                "low_length": low_length,
                "threshold_db": float(threshold),
                "efficiency": summary.efficiency,
                "fairness": summary.fairness,
                "final": summary.final,
            }
            if best_rule is None or candidate["final"] > best_rule["final"]:
                best_rule = candidate
            if low_length == 1:
                threshold_results.append(candidate)

    best_two_threshold_rule = None
    for low_length in (1, 4, 16, 64, 144):
        for middle_length in prefix_lengths[4:-1]:
            for low_threshold in np.arange(-19.0, -8.99, 1.0):
                for high_threshold in np.arange(low_threshold + 1.0, 5.01, 1.0):
                    dynamic = np.where(
                        snr_np < low_threshold,
                        prefix_scores[low_length],
                        np.where(snr_np < high_threshold, prefix_scores[middle_length], full_scores),
                    )
                    summary = summarize_scores(dynamic)
                    candidate = {
                        "low_length": low_length,
                        "middle_length": middle_length,
                        "low_threshold_db": float(low_threshold),
                        "high_threshold_db": float(high_threshold),
                        "efficiency": summary.efficiency,
                        "fairness": summary.fairness,
                        "final": summary.final,
                    }
                    if best_two_threshold_rule is None or candidate["final"] > best_two_threshold_rule["final"]:
                        best_two_threshold_rule = candidate
    print(
        json.dumps(
            {
                "samples": sample_count,
                "checkpoint_step": checkpoint["step"],
                "checkpoint_validation_loss": checkpoint["best_validation"],
                "efficiency": full_summary.efficiency,
                "fairness": full_summary.fairness,
                "final": full_summary.final,
                "best_rule": best_rule,
                "best_two_threshold_rule": best_two_threshold_rule,
                "selected_thresholds": [
                    item
                    for item in threshold_results
                    if item["threshold_db"] in {-18.0, -16.0, -15.0, -14.5, -14.0, -12.0, -10.5}
                ],
                "snr_slices": {
                    f"{low}:{low + 5}": {
                        "mean": float(full_scores[(snr_np >= low) & (snr_np < low + 5)].mean()),
                        "p10": float(np.percentile(full_scores[(snr_np >= low) & (snr_np < low + 5)], 10)),
                    }
                    for low in range(-20, 20, 5)
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

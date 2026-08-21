from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from oppods.data import ChannelMemmap, deterministic_split_indices
from oppods.metrics import per_sample_score, summarize_scores
from oppods.sparse_feedback import SparseDelayMUMIMOLink


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate deterministic sparse task feedback")
    parser.add_argument("--data", type=Path, default=Path("ziliao/data_train/H_train.npz"))
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1176)
    parser.add_argument("--mode-count", type=int, default=6)
    parser.add_argument("--no-wiener", action="store_true")
    parser.add_argument("--wiener-noise-scale", type=float, default=1.0)
    parser.add_argument("--precoder", choices=("mrt", "rzf"), default="mrt")
    parser.add_argument("--regularization-scale", type=float, default=1.0)
    parser.add_argument("--fairness-exponent", type=float)
    parser.add_argument("--receiver-interference-scale", type=float, default=1.0)
    parser.add_argument("--decision-directed-iterations", type=int, default=6)
    parser.add_argument("--central-boost", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    data = ChannelMemmap(args.data)
    validation_indices = deterministic_split_indices(len(data), seed=args.seed)["validation"]
    sample_count = min(args.samples, len(validation_indices))
    indices = validation_indices[:sample_count]
    rng = np.random.default_rng(args.seed)
    bits_np = rng.integers(0, 2, (sample_count, 2, 1152), dtype=np.int8)
    snr_np = rng.uniform(-20.0, 20.0, (sample_count, 2)).astype(np.float32)
    link = SparseDelayMUMIMOLink(
        mode_count=args.mode_count,
        use_wiener=not args.no_wiener,
        wiener_noise_scale=args.wiener_noise_scale,
        precoder=args.precoder,
        regularization_scale=args.regularization_scale,
        fairness_exponent=args.fairness_exponent,
        receiver_interference_scale=args.receiver_interference_scale,
        decision_directed_iterations=args.decision_directed_iterations,
        central_boost=args.central_boost,
    ).to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    prefix_lengths = (1, 4, 16, 64, 144, 288, 432, 576, 720, 864, 1008, 1152)
    prefix_batches: dict[int, list[np.ndarray]] = {length: [] for length in prefix_lengths}
    started = time.perf_counter()
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
    scores = prefix_scores[1152]
    summary = summarize_scores(scores)
    result = {
        "samples": sample_count,
        "mode_count": args.mode_count,
        "use_wiener": not args.no_wiener,
        "wiener_noise_scale": args.wiener_noise_scale,
        "precoder": args.precoder,
        "regularization_scale": args.regularization_scale,
        "fairness_exponent": args.fairness_exponent,
        "receiver_interference_scale": args.receiver_interference_scale,
        "decision_directed_iterations": args.decision_directed_iterations,
        "central_boost": args.central_boost,
        "efficiency": summary.efficiency,
        "fairness": summary.fairness,
        "final": summary.final,
        "elapsed_seconds": time.perf_counter() - started,
        "snr_slices": {},
    }
    best_rule: dict[str, float | int] | None = None
    for low_length in prefix_lengths[:-1]:
        for threshold in np.arange(-20.0, 10.01, 0.5):
            dynamic_scores = np.where(snr_np < threshold, prefix_scores[low_length], scores)
            dynamic_summary = summarize_scores(dynamic_scores)
            candidate: dict[str, float | int] = {
                "low_length": low_length,
                "threshold_db": float(threshold),
                "efficiency": dynamic_summary.efficiency,
                "fairness": dynamic_summary.fairness,
                "final": dynamic_summary.final,
            }
            if best_rule is None or candidate["final"] > best_rule["final"]:
                best_rule = candidate
    result["best_single_threshold_rule"] = best_rule

    best_two_threshold_rule: dict[str, float | int] | None = None
    for low_length in (1, 4, 16, 64, 144):
        for middle_length in prefix_lengths[4:-1]:
            for low_threshold in np.arange(-19.0, -8.99, 1.0):
                for high_threshold in np.arange(low_threshold + 1.0, 5.01, 1.0):
                    dynamic_scores = np.where(
                        snr_np < low_threshold,
                        prefix_scores[low_length],
                        np.where(snr_np < high_threshold, prefix_scores[middle_length], scores),
                    )
                    dynamic_summary = summarize_scores(dynamic_scores)
                    candidate = {
                        "low_length": low_length,
                        "middle_length": middle_length,
                        "low_threshold_db": float(low_threshold),
                        "high_threshold_db": float(high_threshold),
                        "efficiency": dynamic_summary.efficiency,
                        "fairness": dynamic_summary.fairness,
                        "final": dynamic_summary.final,
                    }
                    if best_two_threshold_rule is None or candidate["final"] > best_two_threshold_rule["final"]:
                        best_two_threshold_rule = candidate
    result["best_two_threshold_rule"] = best_two_threshold_rule
    for low in range(-20, 20, 5):
        selected = scores[(snr_np >= low) & (snr_np < low + 5)]
        result["snr_slices"][f"{low}:{low + 5}"] = {
            "mean": float(selected.mean()),
            "p10": float(np.percentile(selected, 10)),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

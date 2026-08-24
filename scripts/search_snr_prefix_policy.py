from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from oppods.metrics import summarize_scores


PREFIXES = (132, 264, 396, 528, 660, 792, 924, 1056)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jointly search a two-region SNR-to-prefix policy from saved exact scores"
    )
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--baseline-threshold", type=float, default=-9.5)
    parser.add_argument("--baseline-prefix", type=int, default=924, choices=PREFIXES)
    parser.add_argument("--threshold-min", type=float, default=-20.0)
    parser.add_argument("--threshold-max", type=float, default=5.0)
    parser.add_argument("--threshold-step", type=float, default=0.5)
    parser.add_argument("--top", type=int, default=20)
    return parser.parse_args()


def policy_scores(
    data: dict[str, np.ndarray], prefix: int, threshold_db: float
) -> np.ndarray:
    return np.where(
        data["snr"] < threshold_db,
        data[f"score_{prefix}"],
        data["score_1056"],
    )


def main() -> None:
    args = parse_args()
    datasets: list[dict[str, np.ndarray]] = []
    for path in args.inputs:
        with np.load(path) as data:
            datasets.append({name: data[name].copy() for name in data.files})

    baseline_summaries = [
        summarize_scores(
            policy_scores(data, args.baseline_prefix, args.baseline_threshold)
        )
        for data in datasets
    ]
    candidates = []
    thresholds = np.arange(
        args.threshold_min,
        args.threshold_max + args.threshold_step * 0.5,
        args.threshold_step,
    )
    for prefix in PREFIXES:
        for threshold in thresholds:
            summaries = [
                summarize_scores(policy_scores(data, prefix, float(threshold)))
                for data in datasets
            ]
            final_deltas = np.asarray(
                [
                    summary.final - baseline.final
                    for summary, baseline in zip(
                        summaries, baseline_summaries, strict=True
                    )
                ]
            )
            fairness_deltas = np.asarray(
                [
                    summary.fairness - baseline.fairness
                    for summary, baseline in zip(
                        summaries, baseline_summaries, strict=True
                    )
                ]
            )
            candidates.append(
                {
                    "prefix": prefix,
                    "threshold_db": float(threshold),
                    "finals": [summary.final for summary in summaries],
                    "fairness": [summary.fairness for summary in summaries],
                    "final_deltas": final_deltas.tolist(),
                    "fairness_deltas": fairness_deltas.tolist(),
                    "mean_final_delta": float(final_deltas.mean()),
                    "min_final_delta": float(final_deltas.min()),
                    "mean_fairness_delta": float(fairness_deltas.mean()),
                    "min_fairness_delta": float(fairness_deltas.min()),
                    "win_count": int(np.count_nonzero(final_deltas > 0.0)),
                }
            )

    stable = [candidate for candidate in candidates if candidate["min_final_delta"] >= 0.0]
    by_mean = sorted(
        candidates,
        key=lambda candidate: (
            candidate["mean_final_delta"],
            candidate["min_final_delta"],
        ),
        reverse=True,
    )
    by_worst = sorted(
        candidates,
        key=lambda candidate: (
            candidate["min_final_delta"],
            candidate["mean_final_delta"],
        ),
        reverse=True,
    )
    stable_by_mean = sorted(
        stable,
        key=lambda candidate: (
            candidate["mean_final_delta"],
            candidate["min_final_delta"],
        ),
        reverse=True,
    )
    print(
        json.dumps(
            {
                "inputs": [str(path) for path in args.inputs],
                "baseline": {
                    "prefix": args.baseline_prefix,
                    "threshold_db": args.baseline_threshold,
                    "finals": [summary.final for summary in baseline_summaries],
                    "fairness": [summary.fairness for summary in baseline_summaries],
                },
                "stable_candidate_count": len(stable),
                "top_stable_mean": stable_by_mean[: args.top],
                "top_mean": by_mean[: args.top],
                "top_worst": by_worst[: args.top],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

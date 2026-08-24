from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from oppods.metrics import summarize_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare paired FATE-MIMO score archives by seed and SNR bin."
    )
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--snr-bin-width", type=float, default=2.5)
    parser.add_argument(
        "--limit",
        type=int,
        help="Compare only the first N user scores from every archive.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.baseline) != len(args.candidate):
        raise ValueError("baseline and candidate archive counts must match")
    if args.snr_bin_width <= 0.0:
        raise ValueError("snr-bin-width must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")

    pair_results: list[dict[str, object]] = []
    all_snrs: list[np.ndarray] = []
    all_deltas: list[np.ndarray] = []
    for baseline_path, candidate_path in zip(
        args.baseline, args.candidate, strict=True
    ):
        with np.load(baseline_path) as baseline, np.load(candidate_path) as candidate:
            selected = slice(None, args.limit)
            baseline_scores = baseline["score"][selected].astype(np.float64)
            candidate_scores = candidate["score"][selected].astype(np.float64)
            if baseline_scores.shape != candidate_scores.shape:
                raise ValueError(f"score shape mismatch: {baseline_path} vs {candidate_path}")
            for key in ("data_index", "snr"):
                if key not in baseline or key not in candidate:
                    raise ValueError(f"paired archives must contain {key!r}")
            if not np.array_equal(
                baseline["data_index"][selected], candidate["data_index"][selected]
            ):
                raise ValueError(f"data_index mismatch: {baseline_path} vs {candidate_path}")
            if not np.array_equal(baseline["snr"][selected], candidate["snr"][selected]):
                raise ValueError(f"snr mismatch: {baseline_path} vs {candidate_path}")

            delta = candidate_scores - baseline_scores
            baseline_summary = summarize_scores(baseline_scores)
            candidate_summary = summarize_scores(candidate_scores)
            pair_results.append(
                {
                    "baseline": str(baseline_path),
                    "candidate": str(candidate_path),
                    "baseline_final": baseline_summary.final,
                    "candidate_final": candidate_summary.final,
                    "final_delta": candidate_summary.final - baseline_summary.final,
                    "efficiency_delta": (
                        candidate_summary.efficiency - baseline_summary.efficiency
                    ),
                    "fairness_delta": (
                        candidate_summary.fairness - baseline_summary.fairness
                    ),
                    "mean_paired_score_delta": float(np.mean(delta)),
                    "positive_scores": int(np.sum(delta > 0.0)),
                    "negative_scores": int(np.sum(delta < 0.0)),
                    "changed_scores": int(np.sum(delta != 0.0)),
                }
            )
            all_snrs.append(baseline["snr"][selected].astype(np.float64))
            all_deltas.append(delta)

    snr = np.concatenate(all_snrs)
    delta = np.concatenate(all_deltas)
    edges = np.arange(-20.0, 20.0 + args.snr_bin_width, args.snr_bin_width)
    snr_bins: list[dict[str, object]] = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        selected = (snr >= low) & (snr < high)
        if not np.any(selected):
            continue
        selected_delta = delta[selected]
        snr_bins.append(
            {
                "low_db": float(low),
                "high_db": float(high),
                "count": int(np.sum(selected)),
                "mean_paired_score_delta": float(np.mean(selected_delta)),
                "positive_scores": int(np.sum(selected_delta > 0.0)),
                "negative_scores": int(np.sum(selected_delta < 0.0)),
            }
        )

    print(
        json.dumps(
            {
                "pairs": pair_results,
                "mean_final_delta": float(
                    np.mean([float(result["final_delta"]) for result in pair_results])
                ),
                "min_final_delta": float(
                    np.min([float(result["final_delta"]) for result in pair_results])
                ),
                "snr_bins": snr_bins,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

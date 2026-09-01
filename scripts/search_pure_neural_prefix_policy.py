from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from oppods.metrics import summarize_scores

PREFIXES = (32, 64, 96, 128, 132, 160, 192, 224, 256, 264, 396, 528, 660, 792, 924, 1008, 1056)
HIGH_PREFIXES = (924, 1008, 1056, 1152)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joint multi-seed search for pure-neural SNR prefix policies")
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--baseline-low", type=float, default=-16.5)
    parser.add_argument("--baseline-middle-prefix", type=int, default=132)
    parser.add_argument("--baseline-high", type=float, default=-14.5)
    parser.add_argument("--baseline-high-prefix", type=int, default=1152, choices=HIGH_PREFIXES)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def policy_scores(
    data: dict[str, np.ndarray],
    low_threshold: float,
    middle_prefix: int,
    high_threshold: float,
    high_prefix: int,
) -> np.ndarray:
    return np.where(
        data["snr"] < low_threshold,
        data["score_1"],
        np.where(
            data["snr"] < high_threshold,
            data[f"score_{middle_prefix}"],
            data[f"score_{high_prefix}"],
        ),
    )


def main() -> None:
    args = parse_args()
    datasets: list[dict[str, np.ndarray]] = []
    for path in args.inputs:
        with np.load(path) as loaded:
            datasets.append({name: loaded[name].copy() for name in loaded.files})
    baselines = [
        summarize_scores(
            policy_scores(
                data,
                args.baseline_low,
                args.baseline_middle_prefix,
                args.baseline_high,
                args.baseline_high_prefix,
            )
        ).final
        for data in datasets
    ]
    candidates: list[dict[str, float | int | list[float]]] = []
    for low_threshold in np.arange(-20.0, -4.99, 0.5):
        for middle_prefix in PREFIXES:
            for high_threshold in np.arange(low_threshold + 0.5, 0.01, 0.5):
                for high_prefix in HIGH_PREFIXES:
                    finals = [
                        summarize_scores(
                            policy_scores(data, low_threshold, middle_prefix, high_threshold, high_prefix)
                        ).final
                        for data in datasets
                    ]
                    deltas = [final - baseline for final, baseline in zip(finals, baselines, strict=True)]
                    candidates.append(
                        {
                            "low_threshold_db": float(low_threshold),
                            "middle_prefix": middle_prefix,
                            "high_threshold_db": float(high_threshold),
                            "high_prefix": high_prefix,
                            "finals": finals,
                            "deltas": deltas,
                            "mean_final": float(np.mean(finals)),
                            "mean_delta": float(np.mean(deltas)),
                            "min_delta": float(np.min(deltas)),
                            "win_count": int(np.sum(np.asarray(deltas) > 0.0)),
                        }
                    )
    by_mean = sorted(candidates, key=lambda item: (item["mean_delta"], item["min_delta"]), reverse=True)
    by_worst = sorted(candidates, key=lambda item: (item["min_delta"], item["mean_delta"]), reverse=True)
    result = {
        "inputs": [str(path) for path in args.inputs],
        "baseline": {
            "low_threshold_db": args.baseline_low,
            "middle_prefix": args.baseline_middle_prefix,
            "high_threshold_db": args.baseline_high,
            "high_prefix": args.baseline_high_prefix,
            "finals": baselines,
            "mean_final": float(np.mean(baselines)),
        },
        "best_mean": by_mean[0],
        "best_worst_case": by_worst[0],
        "top_mean": by_mean[:20],
        "top_worst_case": by_worst[:20],
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from oppods.metrics import summarize_scores

PREFIXES = (132, 264, 396, 528, 660, 792, 924)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jointly search saved strict prefix-score matrices")
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--low-threshold", type=float, default=-15.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = []
    for path in args.inputs:
        with np.load(path) as data:
            datasets.append({name: data[name].copy() for name in data.files})

    baselines = []
    for data in datasets:
        baseline = np.where(
            data["snr"] < args.low_threshold,
            data["score_1"],
            data["score_1056"],
        )
        baselines.append(summarize_scores(baseline).final)

    candidates = []
    for prefix in PREFIXES:
        for high_threshold in np.arange(args.low_threshold + 0.5, 10.01, 0.5):
            finals = []
            for data in datasets:
                selected = np.where(
                    data["snr"] < args.low_threshold,
                    data["score_1"],
                    np.where(
                        data["snr"] < high_threshold,
                        data[f"score_{prefix}"],
                        data["score_1056"],
                    ),
                )
                finals.append(summarize_scores(selected).final)
            deltas = [final - baseline for final, baseline in zip(finals, baselines, strict=True)]
            candidates.append(
                {
                    "middle_prefix": prefix,
                    "high_threshold_db": float(high_threshold),
                    "finals": finals,
                    "deltas": deltas,
                    "mean_delta": float(np.mean(deltas)),
                    "min_delta": float(np.min(deltas)),
                    "win_count": int(np.sum(np.asarray(deltas) > 0.0)),
                }
            )

    by_mean = sorted(candidates, key=lambda item: item["mean_delta"], reverse=True)
    by_worst = sorted(
        candidates,
        key=lambda item: (item["min_delta"], item["mean_delta"]),
        reverse=True,
    )
    print(
        json.dumps(
            {
                "inputs": [str(path) for path in args.inputs],
                "low_threshold_db": args.low_threshold,
                "baselines": baselines,
                "best_mean": by_mean[0],
                "best_worst_case": by_worst[0],
                "top_mean": by_mean[:20],
                "top_worst_case": by_worst[:20],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

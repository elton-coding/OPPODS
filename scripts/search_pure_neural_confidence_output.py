from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from oppods.metrics import summarize_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search confidence-gated 1/132/1152 output for V194")
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--low-boundary", type=float, default=-16.5)
    parser.add_argument("--high-boundary", type=float, default=-14.5)
    parser.add_argument("--quantiles", type=int, default=21)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def fixed_policy(data: dict[str, np.ndarray], low_boundary: float, high_boundary: float) -> np.ndarray:
    return np.where(
        data["snr"] < low_boundary,
        data["score_1"],
        np.where(data["snr"] < high_boundary, data["score_132"], data["score_1152"]),
    )


def threshold_grid(values: np.ndarray, count: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, count)
    return np.unique(np.concatenate(([-np.inf], np.quantile(values, quantiles), [np.inf])))


def main() -> None:
    args = parse_args()
    datasets: list[dict[str, np.ndarray]] = []
    for path in args.inputs:
        with np.load(path) as loaded:
            datasets.append({name: loaded[name].copy() for name in loaded.files})
    required = {
        "snr",
        "score_1",
        "score_132",
        "score_1152",
        "mean_abs_1_132",
        "mean_abs_132_1152",
    }
    for index, data in enumerate(datasets):
        missing = required - data.keys()
        if missing:
            raise ValueError(f"input {index} is missing arrays: {sorted(missing)}")

    low_confidence = np.concatenate(
        [data["mean_abs_1_132"][data["snr"] < args.low_boundary] for data in datasets]
    )
    middle_confidence = np.concatenate(
        [
            data["mean_abs_132_1152"][
                (data["snr"] >= args.low_boundary) & (data["snr"] < args.high_boundary)
            ]
            for data in datasets
        ]
    )
    low_thresholds = threshold_grid(low_confidence, args.quantiles)
    middle_thresholds = threshold_grid(middle_confidence, args.quantiles)
    baseline_finals = [
        summarize_scores(fixed_policy(data, args.low_boundary, args.high_boundary)).final for data in datasets
    ]
    candidates: list[dict[str, float | int | list[float]]] = []
    for low_threshold in low_thresholds:
        for middle_threshold in middle_thresholds:
            finals: list[float] = []
            extension_rates: list[float] = []
            for data in datasets:
                snr = data["snr"]
                low_extend = (snr < args.low_boundary) & (data["mean_abs_1_132"] >= low_threshold)
                middle_band = (snr >= args.low_boundary) & (snr < args.high_boundary)
                middle_extend = middle_band & (data["mean_abs_132_1152"] >= middle_threshold)
                selected = fixed_policy(data, args.low_boundary, args.high_boundary)
                selected = np.where(low_extend, data["score_132"], selected)
                selected = np.where(middle_extend, data["score_1152"], selected)
                finals.append(summarize_scores(selected).final)
                eligible = int(np.sum(snr < args.high_boundary))
                extension_rates.append(float(np.sum(low_extend | middle_extend) / max(eligible, 1)))
            deltas = [final - baseline for final, baseline in zip(finals, baseline_finals, strict=True)]
            candidates.append(
                {
                    "low_confidence_threshold": float(low_threshold),
                    "middle_confidence_threshold": float(middle_threshold),
                    "finals": finals,
                    "deltas": deltas,
                    "mean_delta": float(np.mean(deltas)),
                    "min_delta": float(np.min(deltas)),
                    "win_count": int(np.sum(np.asarray(deltas) > 0.0)),
                    "extension_rates": extension_rates,
                }
            )
    by_mean = sorted(candidates, key=lambda item: (item["mean_delta"], item["min_delta"]), reverse=True)
    by_worst = sorted(candidates, key=lambda item: (item["min_delta"], item["mean_delta"]), reverse=True)
    result = {
        "inputs": [str(path) for path in args.inputs],
        "fixed_policy": {
            "low_boundary_db": args.low_boundary,
            "high_boundary_db": args.high_boundary,
            "finals": baseline_finals,
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

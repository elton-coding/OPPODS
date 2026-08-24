from __future__ import annotations

import argparse
import json

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Greedily search low-correlation Walsh row subsets after truncation."
    )
    parser.add_argument("--length", type=int, default=228)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--top", type=int, default=20)
    return parser.parse_args()


def walsh_vectors(length: int) -> np.ndarray:
    rows = np.arange(1, 256, dtype=np.int16)[:, None]
    positions = np.arange(length, dtype=np.int16)[None]
    parity = np.bitwise_and(rows, positions)
    for shift in (1, 2, 4, 8):
        parity = np.bitwise_xor(parity, np.right_shift(parity, shift))
    bits = np.bitwise_and(parity, 1).astype(np.int16)
    return 1 - 2 * bits


def subset_metrics(indices: list[int], correlation: np.ndarray) -> tuple[int, float, float]:
    matrix = correlation[np.ix_(indices, indices)]
    values = np.abs(matrix[np.triu_indices(len(indices), k=1)]).astype(np.float64)
    return int(np.max(values)), float(np.sqrt(np.mean(values**2))), float(np.mean(values))


def greedy_subset(start: int, count: int, correlation: np.ndarray) -> list[int]:
    selected = [start]
    selected_set = {start}
    current_max = 0
    current_squared_sum = 0
    current_absolute_sum = 0
    while len(selected) < count:
        best: tuple[tuple[int, int, int, int], int, int, int] | None = None
        for candidate in range(correlation.shape[0]):
            if candidate in selected_set:
                continue
            values = np.abs(correlation[candidate, selected]).astype(np.int64)
            candidate_max = max(current_max, int(np.max(values)))
            candidate_squared_sum = current_squared_sum + int(np.sum(values**2))
            candidate_absolute_sum = current_absolute_sum + int(np.sum(values))
            score = (
                candidate_max,
                candidate_squared_sum,
                candidate_absolute_sum,
                candidate,
            )
            if best is None or score < best[0]:
                best = (score, candidate, candidate_squared_sum, candidate_absolute_sum)
        assert best is not None
        _, candidate, current_squared_sum, current_absolute_sum = best
        current_max = max(
            current_max,
            int(np.max(np.abs(correlation[candidate, selected]))),
        )
        selected.append(candidate)
        selected_set.add(candidate)
    return selected


def main() -> None:
    args = parse_args()
    if not 1 <= args.length <= 256:
        raise ValueError("length must be in [1, 256]")
    if not 2 <= args.rows <= 255:
        raise ValueError("rows must be in [2, 255]")

    vectors = walsh_vectors(args.length)
    correlation = vectors.astype(np.int32) @ vectors.astype(np.int32).T
    unique: dict[tuple[int, ...], dict[str, object]] = {}
    for start in range(255):
        indices = greedy_subset(start, args.rows, correlation)
        key = tuple(sorted(indices))
        maximum, rms, mean = subset_metrics(indices, correlation)
        candidate = {
            "rows": [index + 1 for index in indices],
            "max_abs_correlation": maximum,
            "rms_abs_correlation": rms,
            "mean_abs_correlation": mean,
        }
        previous = unique.get(key)
        if previous is None or tuple(candidate["rows"]) < tuple(previous["rows"]):
            unique[key] = candidate

    baseline_indices = list(range(64, 80))
    baseline_max, baseline_rms, baseline_mean = subset_metrics(
        baseline_indices, correlation
    )
    ranked = sorted(
        unique.values(),
        key=lambda candidate: (
            candidate["max_abs_correlation"],
            candidate["rms_abs_correlation"],
            candidate["mean_abs_correlation"],
            candidate["rows"],
        ),
    )
    print(
        json.dumps(
            {
                "length": args.length,
                "row_count": args.rows,
                "baseline_start65": {
                    "rows": list(range(65, 81)),
                    "max_abs_correlation": baseline_max,
                    "rms_abs_correlation": baseline_rms,
                    "mean_abs_correlation": baseline_mean,
                },
                "top": ranked[: args.top],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from oppods.metrics import summarize_scores

PairPredicate = Literal["any-user", "all-user"]


@dataclass(frozen=True)
class PairedScores:
    seed: int
    baseline: np.ndarray
    candidate: np.ndarray
    snr: np.ndarray
    data_index: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search robust joint-UE SNR intervals for a shared candidate."
    )
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--predicate",
        choices=("any-user", "all-user"),
        required=True,
        help="Apply the candidate when any or all UEs lie inside the interval.",
    )
    parser.add_argument("--grid-step", type=float, default=0.25)
    parser.add_argument("--minimum-width", type=float, default=0.25)
    parser.add_argument("--maximum-width", type=float, default=40.0)
    parser.add_argument("--blocks-per-seed", type=int, default=2)
    parser.add_argument("--robust-tolerance", type=float, default=1.0e-7)
    parser.add_argument("--minimum-mean-gain", type=float, default=0.0)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _seed_from_path(path: Path) -> int:
    suffix = path.stem.rsplit("seed", maxsplit=1)
    if len(suffix) != 2:
        raise ValueError(f"cannot infer seed from {path}")
    match = re.match(r"\d+", suffix[1])
    if match is None:
        raise ValueError(f"cannot infer seed from {path}")
    return int(match.group())


def load_pairs(
    baseline_paths: list[Path], candidate_paths: list[Path]
) -> list[PairedScores]:
    if len(baseline_paths) != len(candidate_paths):
        raise ValueError("baseline and candidate archive counts must match")
    result: list[PairedScores] = []
    for baseline_path, candidate_path in zip(
        baseline_paths, candidate_paths, strict=True
    ):
        with np.load(baseline_path) as baseline, np.load(candidate_path) as candidate:
            baseline_scores = baseline["score"].astype(np.float64)
            candidate_scores = candidate["score"].astype(np.float64)
            snr = baseline["snr"].astype(np.float64)
            data_index = baseline["data_index"].astype(np.int64)
            if not (
                baseline_scores.shape
                == candidate_scores.shape
                == snr.shape
                == data_index.shape
            ):
                raise ValueError(
                    f"archive shape mismatch: {baseline_path} vs {candidate_path}"
                )
            for key in ("data_index", "snr"):
                if not np.array_equal(baseline[key], candidate[key]):
                    raise ValueError(
                        f"paired archive {key} mismatch: "
                        f"{baseline_path} vs {candidate_path}"
                    )
            _, counts = np.unique(data_index, return_counts=True)
            if counts.size == 0 or not np.all(counts == 2):
                raise ValueError(f"each data_index must contain exactly two UEs: {baseline_path}")
            baseline_seed = _seed_from_path(baseline_path)
            candidate_seed = _seed_from_path(candidate_path)
            if baseline_seed != candidate_seed:
                raise ValueError(
                    f"seed mismatch: {baseline_path} vs {candidate_path}"
                )
            result.append(
                PairedScores(
                    seed=baseline_seed,
                    baseline=baseline_scores,
                    candidate=candidate_scores,
                    snr=snr,
                    data_index=data_index,
                )
            )
    return result


def _pair_selection(
    pair: PairedScores,
    low_db: float,
    high_db: float,
    predicate: PairPredicate,
) -> tuple[np.ndarray, int]:
    unique_indices, inverse = np.unique(pair.data_index, return_inverse=True)
    grouped_order = np.argsort(inverse, kind="stable")
    group_snr = pair.snr[grouped_order].reshape(unique_indices.size, 2)
    in_interval = (group_snr >= low_db) & (group_snr < high_db)
    if predicate == "any-user":
        selected_pairs = np.any(in_interval, axis=1)
    elif predicate == "all-user":
        selected_pairs = np.all(in_interval, axis=1)
    else:
        raise ValueError(f"unknown pair predicate: {predicate}")
    return selected_pairs[inverse], int(np.sum(selected_pairs))


def evaluate_interval(
    pairs: list[PairedScores],
    low_db: float,
    high_db: float,
    predicate: PairPredicate,
    blocks_per_seed: int,
) -> dict[str, object]:
    seed_results: list[dict[str, object]] = []
    final_deltas: list[float] = []
    fairness_deltas: list[float] = []
    block_final_deltas: list[float] = []
    block_fairness_deltas: list[float] = []
    for pair in pairs:
        selected, selected_pairs = _pair_selection(
            pair, low_db, high_db, predicate
        )
        routed = np.where(selected, pair.candidate, pair.baseline)
        baseline_summary = summarize_scores(pair.baseline)
        routed_summary = summarize_scores(routed)
        final_delta = routed_summary.final - baseline_summary.final
        fairness_delta = routed_summary.fairness - baseline_summary.fairness
        final_deltas.append(final_delta)
        fairness_deltas.append(fairness_delta)

        unique_indices = np.unique(pair.data_index)
        block_results: list[dict[str, float | int]] = []
        for block, block_data_indices in enumerate(
            np.array_split(unique_indices, blocks_per_seed)
        ):
            block_mask = np.isin(pair.data_index, block_data_indices)
            baseline_block = summarize_scores(pair.baseline[block_mask])
            routed_block = summarize_scores(routed[block_mask])
            block_final_delta = routed_block.final - baseline_block.final
            block_fairness_delta = routed_block.fairness - baseline_block.fairness
            block_final_deltas.append(block_final_delta)
            block_fairness_deltas.append(block_fairness_delta)
            block_results.append(
                {
                    "block": block,
                    "selected_pairs": int(
                        np.unique(pair.data_index[block_mask & selected]).size
                    ),
                    "final_delta": block_final_delta,
                    "fairness_delta": block_fairness_delta,
                }
            )

        paired_delta = pair.candidate[selected] - pair.baseline[selected]
        seed_results.append(
            {
                "seed": pair.seed,
                "selected_pairs": selected_pairs,
                "selected_scores": int(np.sum(selected)),
                "changed_scores": int(np.sum(paired_delta != 0.0)),
                "positive": int(np.sum(paired_delta > 0.0)),
                "negative": int(np.sum(paired_delta < 0.0)),
                "score_delta_sum": float(np.sum(paired_delta)),
                "final_delta": final_delta,
                "fairness_delta": fairness_delta,
                "block_results": block_results,
            }
        )

    return {
        "predicate": predicate,
        "low_db": low_db,
        "high_db": high_db,
        "mean_final_delta": float(np.mean(final_deltas)),
        "min_final_delta": float(np.min(final_deltas)),
        "min_fairness_delta": float(np.min(fairness_deltas)),
        "min_block_final_delta": float(np.min(block_final_deltas)),
        "min_block_fairness_delta": float(np.min(block_fairness_deltas)),
        "seed_results": seed_results,
    }


def is_robust(record: dict[str, object], tolerance: float) -> bool:
    return all(
        float(record[key]) >= -tolerance
        for key in (
            "min_final_delta",
            "min_fairness_delta",
            "min_block_final_delta",
            "min_block_fairness_delta",
        )
    )


def ranking_key(record: dict[str, object]) -> tuple[float, float, float]:
    return (
        float(record["mean_final_delta"]),
        float(record["min_final_delta"]),
        float(record["min_block_final_delta"]),
    )


def search_intervals(
    pairs: list[PairedScores],
    *,
    predicate: PairPredicate,
    grid_step: float,
    minimum_width: float,
    maximum_width: float,
    blocks_per_seed: int,
    robust_tolerance: float,
    minimum_mean_gain: float = 0.0,
) -> list[dict[str, object]]:
    if grid_step <= 0.0:
        raise ValueError("grid_step must be positive")
    if minimum_width < grid_step:
        raise ValueError("minimum_width must be at least grid_step")
    if maximum_width < minimum_width:
        raise ValueError("maximum_width must be at least minimum_width")
    if blocks_per_seed < 1:
        raise ValueError("blocks_per_seed must be positive")

    edges = np.arange(-20.0, 20.0 + grid_step * 0.5, grid_step)
    records: list[dict[str, object]] = []
    for low_index, low_db in enumerate(edges[:-1]):
        for high_db in edges[low_index + 1 :]:
            width = float(high_db - low_db)
            if width < minimum_width - 1.0e-9:
                continue
            if width > maximum_width + 1.0e-9:
                break
            record = evaluate_interval(
                pairs,
                float(low_db),
                float(high_db),
                predicate,
                blocks_per_seed,
            )
            if (
                is_robust(record, robust_tolerance)
                and float(record["mean_final_delta"]) > minimum_mean_gain
            ):
                records.append(record)
    records.sort(key=ranking_key, reverse=True)
    return records


def main() -> None:
    args = parse_args()
    pairs = load_pairs(args.baseline, args.candidate)
    records = search_intervals(
        pairs,
        predicate=args.predicate,
        grid_step=args.grid_step,
        minimum_width=args.minimum_width,
        maximum_width=args.maximum_width,
        blocks_per_seed=args.blocks_per_seed,
        robust_tolerance=args.robust_tolerance,
        minimum_mean_gain=args.minimum_mean_gain,
    )
    output = {
        "baseline": [str(path) for path in args.baseline],
        "candidate": [str(path) for path in args.candidate],
        "predicate": args.predicate,
        "grid_step": args.grid_step,
        "blocks_per_seed": args.blocks_per_seed,
        "robust_candidates": len(records),
        "top": records[: args.top],
    }
    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

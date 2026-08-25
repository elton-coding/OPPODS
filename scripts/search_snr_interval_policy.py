from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from oppods.metrics import summarize_scores


@dataclass(frozen=True)
class PairedScores:
    seed: int
    baseline: np.ndarray
    candidate: np.ndarray
    snr: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search robust contiguous SNR intervals for a paired candidate."
    )
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--grid-step", type=float, default=0.25)
    parser.add_argument("--minimum-width", type=float, default=0.25)
    parser.add_argument("--maximum-width", type=float, default=40.0)
    parser.add_argument("--blocks-per-seed", type=int, default=2)
    parser.add_argument("--robust-tolerance", type=float, default=1.0e-7)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _seed_from_path(path: Path) -> int:
    stem = path.stem
    if "seed" not in stem:
        raise ValueError(f"cannot infer seed from {path}")
    suffix = stem.rsplit("seed", maxsplit=1)[1]
    digits = "".join(character for character in suffix if character.isdigit())
    if not digits:
        raise ValueError(f"cannot infer seed from {path}")
    return int(digits)


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
            if baseline_scores.shape != candidate_scores.shape:
                raise ValueError(f"score shape mismatch: {baseline_path} vs {candidate_path}")
            if snr.shape != baseline_scores.shape:
                raise ValueError(f"SNR shape mismatch in {baseline_path}")
            for key in ("data_index", "snr"):
                if not np.array_equal(baseline[key], candidate[key]):
                    raise ValueError(
                        f"paired archive {key} mismatch: {baseline_path} vs {candidate_path}"
                    )
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
                )
            )
    return result


def evaluate_interval(
    pairs: list[PairedScores],
    low_db: float,
    high_db: float,
    blocks_per_seed: int,
) -> dict[str, object]:
    seed_results: list[dict[str, object]] = []
    final_deltas: list[float] = []
    fairness_deltas: list[float] = []
    block_final_deltas: list[float] = []
    block_fairness_deltas: list[float] = []
    for pair in pairs:
        selected = (pair.snr >= low_db) & (pair.snr < high_db)
        routed = np.where(selected, pair.candidate, pair.baseline)
        baseline_summary = summarize_scores(pair.baseline)
        routed_summary = summarize_scores(routed)
        final_delta = routed_summary.final - baseline_summary.final
        fairness_delta = routed_summary.fairness - baseline_summary.fairness
        final_deltas.append(final_delta)
        fairness_deltas.append(fairness_delta)

        block_results: list[dict[str, float | int]] = []
        for block, indices in enumerate(
            np.array_split(np.arange(pair.baseline.size), blocks_per_seed)
        ):
            baseline_block = summarize_scores(pair.baseline[indices])
            routed_block = summarize_scores(routed[indices])
            block_final_delta = routed_block.final - baseline_block.final
            block_fairness_delta = routed_block.fairness - baseline_block.fairness
            block_final_deltas.append(block_final_delta)
            block_fairness_deltas.append(block_fairness_delta)
            block_results.append(
                {
                    "block": block,
                    "final_delta": block_final_delta,
                    "fairness_delta": block_fairness_delta,
                }
            )

        paired_delta = pair.candidate[selected] - pair.baseline[selected]
        seed_results.append(
            {
                "seed": pair.seed,
                "selected": int(np.sum(selected)),
                "positive": int(np.sum(paired_delta > 0.0)),
                "negative": int(np.sum(paired_delta < 0.0)),
                "score_delta_sum": float(np.sum(paired_delta)),
                "final_delta": final_delta,
                "fairness_delta": fairness_delta,
                "block_results": block_results,
            }
        )

    return {
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
    grid_step: float,
    minimum_width: float,
    maximum_width: float,
    blocks_per_seed: int,
    robust_tolerance: float,
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
                blocks_per_seed,
            )
            if is_robust(record, robust_tolerance):
                records.append(record)
    records.sort(key=ranking_key, reverse=True)
    return records


def main() -> None:
    args = parse_args()
    pairs = load_pairs(args.baseline, args.candidate)
    records = search_intervals(
        pairs,
        grid_step=args.grid_step,
        minimum_width=args.minimum_width,
        maximum_width=args.maximum_width,
        blocks_per_seed=args.blocks_per_seed,
        robust_tolerance=args.robust_tolerance,
    )
    output = {
        "baseline": [str(path) for path in args.baseline],
        "candidate": [str(path) for path in args.candidate],
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

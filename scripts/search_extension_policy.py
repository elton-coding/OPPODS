from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

from oppods.metrics import summarize_scores


@dataclass(frozen=True)
class SeedData:
    seed: int
    baseline_scores: np.ndarray
    positions: np.ndarray
    fallback_scores: np.ndarray
    extension_scores: np.ndarray
    snr: np.ndarray
    metrics: dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search robust confidence gates for the 924-to-1056 bit middle-SNR extension."
    )
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--diagnostics", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quantiles", type=int, default=129)
    parser.add_argument("--top", type=int, default=20)
    return parser.parse_args()


def _derived_metrics(archive: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    eps = 1.0e-9
    metrics = {
        key: archive[key].astype(np.float64)
        for key in (
            "mean_abs",
            "median_abs",
            "q25_abs",
            "q75_abs",
            "std_abs",
            "mean_clipped_0p25",
            "mean_clipped_0p5",
            "mean_clipped_1p0",
        )
    }
    metrics.update(
        {
            "mean_over_std": metrics["mean_abs"] / (metrics["std_abs"] + eps),
            "median_over_mean": metrics["median_abs"] / (metrics["mean_abs"] + eps),
            "q25_over_mean": metrics["q25_abs"] / (metrics["mean_abs"] + eps),
            "negative_cv": -metrics["std_abs"] / (metrics["mean_abs"] + eps),
            "mean_minus_std": metrics["mean_abs"] - metrics["std_abs"],
            "q25_plus_median": metrics["q25_abs"] + metrics["median_abs"],
        }
    )
    return metrics


def load_seed_data(baseline_paths: list[Path], diagnostic_paths: list[Path]) -> list[SeedData]:
    if len(baseline_paths) != len(diagnostic_paths):
        raise ValueError("baseline and diagnostics archive counts must match")
    result: list[SeedData] = []
    for baseline_path, diagnostic_path in zip(baseline_paths, diagnostic_paths, strict=True):
        with np.load(baseline_path) as baseline, np.load(diagnostic_path) as diagnostics:
            positions = diagnostics["score_position"].astype(np.int64)
            scores = baseline["score"].astype(np.float64)
            fallback = diagnostics["fallback_score"].astype(np.float64)
            if positions.size != np.unique(positions).size:
                raise ValueError(f"duplicate score positions in {diagnostic_path}")
            if positions.size and (positions.min() < 0 or positions.max() >= scores.size):
                raise ValueError(f"invalid score position in {diagnostic_path}")
            maximum_error = float(np.max(np.abs(scores[positions] - fallback), initial=0.0))
            if maximum_error > 1.0e-4:
                raise ValueError(
                    f"fallback scores do not reconstruct {baseline_path}: max error {maximum_error}"
                )
            scores[positions] = fallback
            seed_text = diagnostic_path.stem.rsplit("seed", maxsplit=1)[-1]
            result.append(
                SeedData(
                    seed=int(seed_text),
                    baseline_scores=scores,
                    positions=positions,
                    fallback_scores=fallback,
                    extension_scores=diagnostics["extension_score"].astype(np.float64),
                    snr=diagnostics["snr"].astype(np.float64),
                    metrics=_derived_metrics(diagnostics),
                )
            )
    return result


def threshold_grid(values: np.ndarray, quantiles: int, extra: float | None = None) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("cannot build a threshold grid from no finite values")
    grid = np.quantile(finite, np.linspace(0.0, 1.0, quantiles))
    grid = np.concatenate((grid, [np.nextafter(finite.max(), np.inf)]))
    if extra is not None:
        grid = np.concatenate((grid, [extra]))
    return np.unique(grid)


def evaluate_policy(
    seed_data: list[SeedData],
    selections: list[np.ndarray],
) -> dict[str, object]:
    seed_results: list[dict[str, object]] = []
    final_deltas: list[float] = []
    for data, selected in zip(seed_data, selections, strict=True):
        candidate_scores = data.baseline_scores.copy()
        candidate_scores[data.positions] = np.where(
            selected, data.extension_scores, data.fallback_scores
        )
        baseline_summary = summarize_scores(data.baseline_scores)
        candidate_summary = summarize_scores(candidate_scores)
        final_delta = candidate_summary.final - baseline_summary.final
        final_deltas.append(final_delta)
        selected_delta = data.extension_scores[selected] - data.fallback_scores[selected]
        seed_results.append(
            {
                "seed": data.seed,
                "final": candidate_summary.final,
                "final_delta": final_delta,
                "efficiency": candidate_summary.efficiency,
                "fairness": candidate_summary.fairness,
                "selected": int(np.sum(selected)),
                "positive": int(np.sum(selected_delta > 0.0)),
                "negative": int(np.sum(selected_delta < 0.0)),
                "mean_selected_score_delta": (
                    float(np.mean(selected_delta)) if selected_delta.size else 0.0
                ),
            }
        )
    return {
        "mean_final_delta": float(np.mean(final_deltas)),
        "min_final_delta": float(np.min(final_deltas)),
        "seed_results": seed_results,
    }


def is_robust(record: dict[str, object]) -> bool:
    return float(record["min_final_delta"]) >= -1.0e-12


def ranking_key(record: dict[str, object]) -> tuple[float, float]:
    return float(record["mean_final_delta"]), float(record["min_final_delta"])


def main() -> None:
    args = parse_args()
    if args.quantiles < 3:
        raise ValueError("quantiles must be at least 3")
    seed_data = load_seed_data(args.baseline, args.diagnostics)
    metric_names = list(seed_data[0].metrics)
    if any(list(data.metrics) != metric_names for data in seed_data[1:]):
        raise ValueError("diagnostic metric keys do not match")

    baseline_results = []
    for data in seed_data:
        summary = summarize_scores(data.baseline_scores)
        baseline_results.append({"seed": data.seed, **summary.__dict__})

    current = evaluate_policy(
        seed_data,
        [data.metrics["mean_abs"] >= 0.3 for data in seed_data],
    )
    current.update({"metric": "mean_abs", "threshold": 0.3})

    global_candidates: list[dict[str, object]] = []
    best_global_by_metric: dict[str, dict[str, object]] = {}
    for metric_name in metric_names:
        pooled = np.concatenate([data.metrics[metric_name] for data in seed_data])
        extra = 0.3 if metric_name == "mean_abs" else None
        for threshold in threshold_grid(pooled, args.quantiles, extra):
            record = evaluate_policy(
                seed_data,
                [data.metrics[metric_name] >= threshold for data in seed_data],
            )
            record.update({"metric": metric_name, "threshold": float(threshold)})
            if is_robust(record):
                global_candidates.append(record)
                previous = best_global_by_metric.get(metric_name)
                if previous is None or ranking_key(record) > ranking_key(previous):
                    best_global_by_metric[metric_name] = record

    bin_edges = np.arange(-20.0, -2.5 + 1.0e-9, 2.5)
    binned_candidates: list[dict[str, object]] = []
    for metric_name, global_record in best_global_by_metric.items():
        thresholds = np.full(bin_edges.size - 1, float(global_record["threshold"]))

        def selections_for(
            candidate_thresholds: np.ndarray,
            selected_metric: str = metric_name,
        ) -> list[np.ndarray]:
            selections = []
            for data in seed_data:
                bin_index = np.searchsorted(bin_edges, data.snr, side="right") - 1
                bin_index = np.clip(bin_index, 0, candidate_thresholds.size - 1)
                selections.append(data.metrics[selected_metric] >= candidate_thresholds[bin_index])
            return selections

        best_record = evaluate_policy(seed_data, selections_for(thresholds))
        for _ in range(4):
            changed = False
            for bin_index, (low, high) in enumerate(
                pairwise(bin_edges)
            ):
                pooled_bin = np.concatenate(
                    [
                        data.metrics[metric_name][(data.snr >= low) & (data.snr < high)]
                        for data in seed_data
                    ]
                )
                if pooled_bin.size == 0:
                    continue
                candidates = threshold_grid(
                    pooled_bin,
                    max(17, args.quantiles // 4),
                    thresholds[bin_index],
                )
                local_threshold = thresholds[bin_index]
                local_record = best_record
                for candidate_threshold in candidates:
                    trial = thresholds.copy()
                    trial[bin_index] = candidate_threshold
                    record = evaluate_policy(seed_data, selections_for(trial))
                    if is_robust(record) and ranking_key(record) > ranking_key(local_record):
                        local_threshold = candidate_threshold
                        local_record = record
                if local_threshold != thresholds[bin_index]:
                    thresholds[bin_index] = local_threshold
                    best_record = local_record
                    changed = True
            if not changed:
                break
        best_record.update(
            {
                "metric": metric_name,
                "bin_edges_db": bin_edges.tolist(),
                "thresholds": thresholds.tolist(),
            }
        )
        binned_candidates.append(best_record)

    global_candidates.sort(key=ranking_key, reverse=True)
    binned_candidates.sort(key=ranking_key, reverse=True)
    result = {
        "baseline": baseline_results,
        "current_v124": current,
        "search": {
            "quantiles": args.quantiles,
            "robust_constraint": "minimum final delta across seeds >= 0",
            "snr_bin_width_db": 2.5,
        },
        "top_global": global_candidates[: args.top],
        "top_binned": binned_candidates[: args.top],
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

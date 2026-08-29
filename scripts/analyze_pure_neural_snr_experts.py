from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path

import numpy as np

EDGES_DB = np.arange(-20.0, 20.1, 5.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare paired exact scores for the V191 SNR expert bank")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def summarize(scores: np.ndarray) -> dict[str, float]:
    efficiency = float(np.mean(scores))
    fairness = float(np.percentile(scores, 10))
    return {
        "efficiency": efficiency,
        "fairness": fairness,
        "final": 0.7 * efficiency + 0.3 * fairness,
    }


def interval_mask(values: np.ndarray, low_db: float, high_db: float) -> np.ndarray:
    if high_db == EDGES_DB[-1]:
        return (values >= low_db) & (values <= high_db)
    return (values >= low_db) & (values < high_db)


def interval_rows(
    baseline_scores: np.ndarray,
    candidate_scores: np.ndarray,
    routing_snr: np.ndarray,
) -> list[dict[str, float | int | list[float]]]:
    rows: list[dict[str, float | int | list[float]]] = []
    for low_db, high_db in pairwise(EDGES_DB):
        selected = interval_mask(routing_snr, low_db, high_db)
        baseline = summarize(baseline_scores[selected])
        candidate = summarize(candidate_scores[selected])
        rows.append(
            {
                "snr_interval_db": [float(low_db), float(high_db)],
                "count": int(selected.sum()),
                "baseline_efficiency": baseline["efficiency"],
                "candidate_efficiency": candidate["efficiency"],
                "mean_delta": float(np.mean(candidate_scores[selected] - baseline_scores[selected])),
                "baseline_p10": baseline["fairness"],
                "candidate_p10": candidate["fairness"],
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    baseline = np.load(args.baseline)
    candidate = np.load(args.candidate)
    for key in ("snr", "data_index"):
        if not np.array_equal(baseline[key], candidate[key]):
            raise ValueError(f"paired files differ in {key}")
    baseline_scores = baseline["score"].astype(np.float64)
    candidate_scores = candidate["score"].astype(np.float64)
    snr = baseline["snr"].astype(np.float64)
    if baseline_scores.shape != candidate_scores.shape or baseline_scores.shape != snr.shape:
        raise ValueError("score and SNR shapes differ")
    if len(scores_by_sample := baseline_scores.reshape(-1, 2)) * 2 != len(baseline_scores):
        raise ValueError("expected two UE scores per sample")
    candidate_by_sample = candidate_scores.reshape(-1, 2)
    snr_by_sample = snr.reshape(-1, 2)
    pair_min_snr = np.repeat(snr_by_sample.min(axis=1), 2)
    result = {
        "baseline": summarize(baseline_scores),
        "candidate": summarize(candidate_scores),
        "final_delta": summarize(candidate_scores)["final"] - summarize(baseline_scores)["final"],
        "per_ue_snr": interval_rows(baseline_scores, candidate_scores, snr),
        "per_pair_min_snr": interval_rows(
            scores_by_sample.reshape(-1),
            candidate_by_sample.reshape(-1),
            pair_min_snr,
        ),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()

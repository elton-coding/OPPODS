"""Compare fixed-partition evaluations, resampling paired UE channels together."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def metrics(scores: np.ndarray) -> dict[str, float]:
    efficiency = float(scores.mean())
    fairness = float(np.percentile(scores, 10))
    return {"efficiency": efficiency, "p10": fairness, "final": .7 * efficiency + .3 * fairness}


def compare(baseline_paths: list[Path], candidate_paths: list[Path], repeats: int = 2000) -> dict:
    if not baseline_paths or len(baseline_paths) != len(candidate_paths):
        raise ValueError("need matching nonempty baseline and candidate lists")
    baseline, candidate, records = [], [], []
    reference_ids = None
    for bp, cp in zip(baseline_paths, candidate_paths, strict=True):
        with np.load(bp) as b, np.load(cp) as c:
            for key in ("data_index", "snr", "split_seed", "noise_seed", "test_offset"):
                if not np.array_equal(b[key], c[key]):
                    raise ValueError(f"unpaired {key}: {bp} vs {cp}")
            ids = b["data_index"].reshape(-1, 2)
            if not np.array_equal(ids[:, 0], ids[:, 1]):
                raise ValueError("consecutive rows must be the two UEs of the same channel")
            if reference_ids is not None and not np.array_equal(ids, reference_ids):
                raise ValueError("noise replicas must use identical channel IDs")
            reference_ids = ids
            bs, cs = b["score"].astype(np.float64), c["score"].astype(np.float64)
            baseline.append(bs.reshape(-1, 2))
            candidate.append(cs.reshape(-1, 2))
            bm, cm = metrics(bs), metrics(cs)
            snr = b["snr"]
            pair_min = snr.reshape(-1, 2).min(axis=1).repeat(2)
            records.append({
                "baseline_file": str(bp), "candidate_file": str(cp),
                "noise_seed": int(b["noise_seed"]), "baseline": bm, "candidate": cm,
                "delta": {key: cm[key] - bm[key] for key in bm},
                "mean_score_delta_by_own_snr": {
                    f"[{lo},{lo + 5})": float((cs - bs)[(snr >= lo) & (snr < lo + 5)].mean())
                    for lo in range(-20, 20, 5)
                },
                "pair_min_ge_minus10_max_absolute_score_delta": float(np.max(
                    np.abs((cs - bs)[pair_min >= -10]), initial=0,
                )),
            })
    barray, carray = np.stack(baseline), np.stack(candidate)
    rng = np.random.default_rng(227)
    boot = []
    for _ in range(repeats):
        index = rng.integers(0, barray.shape[1], barray.shape[1])
        delta = [metrics(c[index])["final"] - metrics(b[index])["final"]
                 for b, c in zip(barray, carray, strict=True)]
        boot.append(float(np.mean(delta)))
    return {
        "protocol": "fixed split1176; channel-cluster paired bootstrap shared across noise replicas",
        "per_seed": records,
        "baseline_mean_final": float(np.mean([r["baseline"]["final"] for r in records])),
        "candidate_mean_final": float(np.mean([r["candidate"]["final"] for r in records])),
        "mean_delta": float(np.mean([r["delta"]["final"] for r in records])),
        "bootstrap_repeats": repeats,
        "paired_delta_95_percentile_interval": np.percentile(boot, [2.5, 97.5]).tolist(),
        "caveat": "finite-channel uncertainty only; not adjusted for repeated model selection or online shift",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", nargs="+", type=Path, required=True)
    parser.add_argument("--candidate", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.baseline, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

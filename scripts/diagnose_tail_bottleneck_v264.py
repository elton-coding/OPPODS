"""Read-only post-selection diagnostics of completed paired caches; no model fitting."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from audit_pure_neural_candidate import score_paths
from compare_paired_holdout import metrics
from run_frozen_confirmation_v258 import checked_caches
from run_pure_neural_lr_v237 import ROOT, fingerprints
from run_storage_factorial_v254 import write_new

LABELS = {
    "V250": "v250_eight_rms_72k",
    "V257": "v257_eight_shared_prefix8_rms_72k",
    "V258": "v258_eight_rms_endtoend_72k",
    "V261": "v261_eight_rms_hard_rank_72k",
    "V262": "v262_eight_rms_shared_endtoend_72k",
}
SEEDS = [22701, 22702, 22703]
OUTPUT = ROOT / "benchmarks/v264_completed_tail_diagnostic.json"


def bins(snr):
    return np.clip(np.searchsorted(np.arange(-15, 20, 5), snr, side="right"), 0, 7)


def summarize(reference, candidates, mask, tail, near):
    count = int(mask.sum())
    if not count:
        return {"rows": 0}
    return {
        "rows": count,
        "share_of_all_rows": float(mask.mean()),
        "reference_mean": float(reference[mask].mean()),
        "reference_bottom10_rows": int((mask & tail).sum()),
        "share_of_reference_bottom10": float((mask & tail).sum() / max(1, tail.sum())),
        "reference_near_p10_rows": int((mask & near).sum()),
        "share_of_reference_near_p10": float((mask & near).sum() / max(1, near.sum())),
        "candidates": {name: {
            "mean_delta": float((value-reference)[mask].mean()),
            "global_efficiency_contribution": float((value-reference)[mask].sum() / reference.size),
            "mean_delta_on_fixed_reference_tail": float((value-reference)[mask & tail].mean())
                if (mask & tail).any() else None,
            "mean_delta_on_fixed_reference_near_p10": float((value-reference)[mask & near].mean())
                if (mask & near).any() else None,
        } for name, value in candidates.items()},
    }


def analyze(snr, scores):
    if set(scores) != set(LABELS) or any(v.shape != snr.shape for v in scores.values()):
        raise ValueError("need aligned five completed models")
    reference = scores["V257"]
    q10 = np.percentile(reference, 10)
    tail, near = reference <= q10, np.abs(reference-q10) <= 1.0
    own = bins(snr)
    partner = bins(snr.reshape(-1, 2)[:, ::-1].reshape(-1))
    minimum = np.minimum(own, partner)
    maximum = np.maximum(own, partner)
    candidates = {k: v for k, v in scores.items() if k != "V257"}
    groupings = {
        "own_snr": {str(i): own == i for i in range(8)},
        "pair_min_expert": {str(i): minimum == i for i in range(8)},
        "own_partner_snr": {f"{i},{j}": (own == i) & (partner == j) for i in range(8) for j in range(8)},
        "pair_min_max_snr": {f"{i},{j}": (minimum == i) & (maximum == j)
                             for i in range(8) for j in range(i, 8)},
    }
    return {
        "reference_p10": float(q10), "model_metrics": {k: metrics(v) for k, v in scores.items()},
        "reference_bottom10_rows_including_ties": int(tail.sum()),
        "reference_near_p10_rows": int(near.sum()),
        "slices": {kind: {key: summarize(reference, candidates, mask, tail, near)
                           for key, mask in masks.items()} for kind, masks in groupings.items()},
    }


def main():
    if OUTPUT.exists():
        raise FileExistsError("diagnostic already exists; do not overwrite")
    checked_caches(list(LABELS.values()), offset=2000, seeds=SEEDS)
    paths = {name: score_paths(ROOT / "benchmarks", label, SEEDS, 2000) for name, label in LABELS.items()}
    records, all_snr = [], []
    all_scores = {name: [] for name in LABELS}
    for i, seed in enumerate(SEEDS):
        scores, snr = {}, None
        for name, group in paths.items():
            with np.load(group[i], allow_pickle=False) as cache:
                current = cache["snr"].astype(np.float64)
                if snr is not None and not np.array_equal(snr, current):
                    raise ValueError("unpaired SNR")
                snr = current
                scores[name] = cache["score"].astype(np.float64)
            all_scores[name].append(scores[name])
        all_snr.append(snr)
        records.append({"seed": seed, **analyze(snr, scores)})
    # Pooled masks retain each noise replica's own global P10; never pool percentiles as official score.
    reference = np.concatenate(all_scores["V257"])
    tail = np.concatenate([v <= np.percentile(v, 10) for v in all_scores["V257"]])
    near = np.concatenate([np.abs(v-np.percentile(v, 10)) <= 1 for v in all_scores["V257"]])
    snr = np.concatenate(all_snr)
    own = bins(snr)
    partner = bins(snr.reshape(-1, 2)[:, ::-1].reshape(-1))
    candidates = {k: np.concatenate(v) for k, v in all_scores.items() if k != "V257"}
    pooled = {"own_snr": {}, "pair_min_expert": {}, "own_partner_snr": {}}
    for i in range(8):
        pooled["own_snr"][str(i)] = summarize(reference, candidates, own == i, tail, near)
        pooled["pair_min_expert"][str(i)] = summarize(reference, candidates, np.minimum(own, partner) == i, tail, near)
        for j in range(8):
            pooled["own_partner_snr"][f"{i},{j}"] = summarize(reference, candidates, (own == i) & (partner == j), tail, near)
    inputs = [path for group in paths.values() for path in group]
    inputs += [Path(__file__), ROOT / "scripts/run_frozen_confirmation_v258.py"]
    result = {"labels": LABELS, "input_sha256": fingerprints(inputs), "per_seed": records,
              "pooled_descriptive_slices": pooled, "snr_edges": list(range(-20, 21, 5)),
              "diagnostic_only": True, "model_updates": False, "new_evaluation": False,
              "caveats": ["Previously viewed selection caches; no new confirmation evidence.",
                          "Noise replicas share 2000 channel IDs; 12000 UE rows are not independent channels.",
                          "Tail membership is fixed by V257; conditional improvements are not global P10 gains.",
                          "Reference-tail selection can induce regression-to-mean effects; descriptive, not causal.",
                          "Scores alone cannot measure gradient conflict, channel estimation error, or overfitting."]}
    write_new(OUTPUT, result)
    print(json.dumps({"output": str(OUTPUT), "slices": pooled["own_snr"], "experts": pooled["pair_min_expert"]}))


if __name__ == "__main__":
    main()

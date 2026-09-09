"""Frozen V250 versus V242E; conditional cohort confirmation, not global blindness."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

from audit_pure_neural_candidate import fingerprint, score_paths
from run_pure_neural_long_budget_v250 import check_prefix, data_record, require_report
from run_pure_neural_lr_v237 import ROOT, fingerprints
from run_pure_neural_rms_budget_v242 import require_audit

SEEDS = [42701, 42702, 42703]
OFFSET = 6000
PROTOCOL = {"split_seed": 1176, "test_offset": OFFSET, "samples": 2000, "noise_seeds": SEEDS}
SOURCES = [
    ("v242_eight_rms_36k", "v242e_confirm_v250", "artifacts/pure_neural_v242/eight/steps36000", 36000),
    ("v250_eight_rms_72k", "v250_confirm", "artifacts/pure_neural_v250/eight/steps72000", 72000),
]
PLAN_PATH = ROOT / "benchmarks/v250_confirmation_plan.json"
RESULT_PATH = ROOT / "benchmarks/v250_confirmation_decision.json"
INVENTORY = "benchmarks/confirmation_channel_inventory_20260909_0925.json"


def command(label, directory, baseline):
    return [sys.executable, "-u", "scripts/audit_pure_neural_candidate.py", "--submission", directory,
            "--label", label, "--baseline-label", baseline, "--test-offset", str(OFFSET),
            "--samples", "2000", "--noise-seeds", *map(str, SEEDS)]


def make_plan():
    sources, reports = [], []
    for source_label, label, directory, steps in SOURCES:
        source_path = ROOT / f"benchmarks/{source_label}_audit_offset2000.json"
        report = json.loads(source_path.read_text(encoding="utf-8"))
        require_audit(report, source_label, steps)
        require_report(report["training_report"], steps)
        files = fingerprint(ROOT / directory)
        if files != report["files"]:
            raise RuntimeError("model changed since completed source audit")
        reports.append(report)
        sources.append({"source_audit": str(source_path.relative_to(ROOT)), "label": label,
                        "directory": directory, "requested_steps": steps, "files": files})
    prefix = check_prefix(reports[1]["training_report"], reports[0]["training_report"])
    if not prefix["matched"]:
        raise RuntimeError("V250 source trajectory did not replicate V242E")
    selection = reports[1]["comparisons"][SOURCES[0][0]]
    if (min(row["delta"]["final"] for row in selection["per_seed"]) <= 0
            or selection["paired_delta_95_percentile_interval"][0] <= 0):
        raise ValueError("candidate did not pass fixed selection gate")
    inventory = json.loads((ROOT / INVENTORY).read_text(encoding="utf-8"))
    window = next(row for row in inventory["windows"] if row["test_offset"] == OFFSET)
    if window["samples"] != 2000 or any(window[key] for key in (
            "train1176_overlap", "validation1176_overlap", "current_offset2000_audit_overlap")):
        raise ValueError("confirmation window intersects current training/validation/selection")
    paths = [ROOT / name for name in (
        "scripts/run_frozen_confirmation_v250.py", "scripts/audit_pure_neural_candidate.py",
        "scripts/evaluate_submission.py", "scripts/compare_paired_holdout.py",
        "scripts/run_pure_neural_long_budget_v250.py", "scripts/run_pure_neural_rms_budget_v242.py",
        "scripts/run_pure_neural_lr_v237.py", "scripts/run_pure_neural_budget_v240.py",
        "src/oppods/data.py", "ziliao/data_train/H_train.npz", INVENTORY,
        "benchmarks/v250_eight_rms_72k_prefix_check.json",
        "artifacts/pure_neural_v250/eight/execution_plan.json")]
    paths += [ROOT / row["source_audit"] for row in sources]
    return {"sources": sources, "input_sha256": fingerprints(paths), "dataset": data_record(),
            "prefix": prefix, "protocol": PROTOCOL,
            "commands": [command(SOURCES[0][1], SOURCES[0][2], ""),
                         command(SOURCES[1][1], SOURCES[1][2], SOURCES[0][1])],
            "criteria": "all three paired total deltas > 0 and channel-bootstrap 95% lower bound > 0",
            "scope": "unused for this V230+ cohort selection; offset4000 deliberately not reused",
            "historical_inventory_window": window, "whole_project_blindness_certified": False,
            "ancestor_training_provenance_certified": False, "online_confirmation": False}


def decision(audit):
    if audit.get("label") != SOURCES[1][1] or audit.get("protocol") != PROTOCOL:
        raise ValueError("wrong frozen confirmation protocol")
    comparison = audit["comparisons"][SOURCES[0][1]]
    rows = comparison["per_seed"]
    if [row["noise_seed"] for row in rows] != SEEDS:
        raise ValueError("missing, duplicated, or reordered confirmation noise seeds")
    deltas = [row["delta"]["final"] for row in rows]
    interval = comparison["paired_delta_95_percentile_interval"]
    values = [*deltas, *interval, comparison["mean_delta"], audit["exact_mean_final"]]
    if len(interval) != 2 or not all(math.isfinite(value) for value in values) or interval[0] > interval[1]:
        raise ValueError("non-finite or invalid confirmation statistics")
    return {"supports_current_cohort_promotion": min(deltas) > 0 and interval[0] > 0,
            "candidate_exact_mean_final": audit["exact_mean_final"],
            "paired_mean_delta": comparison["mean_delta"], "paired_delta_95_interval": interval,
            "per_seed_deltas": deltas, "per_seed_metric_deltas": [row["delta"] for row in rows],
            "whole_project_blindness_certified": False, "ancestor_training_provenance_certified": False,
            "online_confirmation": False,
            "caveat": "conditional cohort confirmation only; historical overlap and ancestor provenance gaps remain"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    outputs = [ROOT / f"benchmarks/{label}_audit_offset{OFFSET}.json" for _, label, _, _ in SOURCES]
    outputs += [path for _, label, _, _ in SOURCES
                for path in score_paths(ROOT / "benchmarks", label, SEEDS, OFFSET)]
    outputs.append(RESULT_PATH)
    if any(path.exists() for path in outputs):
        raise FileExistsError("confirmation output exists: inspect; never repeat or overwrite")
    plan = make_plan()
    if args.register_only:
        with PLAN_PATH.open("x", encoding="utf-8") as stream:
            json.dump(plan, stream, indent=2)
        print(json.dumps({"registered": str(PLAN_PATH), "evaluation_started": False}), flush=True)
        return
    registered = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if registered != plan:
        raise RuntimeError("freeze differs from registered plan")
    lock = ROOT / "artifacts/v250_confirmation.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        for cmd in plan["commands"]:
            subprocess.run(cmd, cwd=ROOT, check=True)
            if make_plan() != registered:
                raise RuntimeError("frozen inputs changed during confirmation")
        candidate = json.loads(outputs[1].read_text(encoding="utf-8"))
        result = decision(candidate)
        with RESULT_PATH.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
        print(json.dumps(result), flush=True)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()

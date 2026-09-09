"""Frozen V242E versus V240C on a preregistered, current-cohort unused window.

This is not whole-project blind: V243 found historical channel overlaps and a
partial ancestor-provenance gap. No candidate fitting or window selection here.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from audit_pure_neural_candidate import fingerprint, score_paths
from run_pure_neural_lr_v237 import ROOT, fingerprints
from run_pure_neural_rms_budget_v242 import require_audit

SEEDS = [32701, 32702, 32703]
OFFSET = 4000
SOURCES = [("v240_two_36k", "v240c_confirm_v242e", "artifacts/pure_neural_v240/two/steps36000"),
           ("v242_eight_rms_36k", "v242e_confirm", "artifacts/pure_neural_v242/eight/steps36000")]
PLAN_PATH = ROOT / "benchmarks/v242e_confirmation_plan.json"
RESULT_PATH = ROOT / "benchmarks/v242e_confirmation_decision.json"


def command(label, directory, baseline):
    return [sys.executable, "-u", "scripts/audit_pure_neural_candidate.py", "--submission", directory,
            "--label", label, "--baseline-label", baseline, "--test-offset", str(OFFSET),
            "--samples", "2000", "--noise-seeds", *map(str, SEEDS)]


def make_plan():
    sources = []
    for source_label, label, directory in SOURCES:
        source_path = ROOT / f"benchmarks/{source_label}_audit_offset2000.json"
        report = json.loads(source_path.read_text(encoding="utf-8"))
        require_audit(report, source_label, 36000)
        files = fingerprint(ROOT / directory)
        if files != report["files"]:
            raise RuntimeError("model changed since original completed audit")
        sources.append({"source_audit": str(source_path.relative_to(ROOT)), "label": label,
                        "directory": directory, "files": files})
    inputs = [ROOT / name for name in ("scripts/run_frozen_confirmation_v242e.py",
              "scripts/audit_pure_neural_candidate.py", "scripts/evaluate_submission.py",
              "scripts/compare_paired_holdout.py", "scripts/run_pure_neural_rms_budget_v242.py",
              "scripts/run_pure_neural_lr_v237.py", "benchmarks/confirmation_channel_inventory_20260909_0925.json")]
    inputs += [ROOT / source["source_audit"] for source in sources]
    return {"sources": sources, "input_sha256": fingerprints(inputs),
            "protocol": {"split_seed": 1176, "test_offset": OFFSET, "samples": 2000, "noise_seeds": SEEDS},
            "commands": [command(SOURCES[0][1], SOURCES[0][2], ""),
                         command(SOURCES[1][1], SOURCES[1][2], SOURCES[0][1])],
            "criteria": "all three paired total deltas > 0 and channel-bootstrap 95% lower bound > 0",
            "scope": "unused for this V230+ cohort's offset2000 selection, NOT whole-project unseen",
            "historical_channel_overlap": 1779,
            "whole_project_blindness_certified": False,
            "ancestor_training_provenance_certified": False,
            "online_confirmation": False}


def decision(audit):
    if audit.get("label") != SOURCES[1][1] or audit.get("protocol") != {
            "split_seed": 1176, "test_offset": OFFSET, "samples": 2000, "noise_seeds": SEEDS}:
        raise ValueError("wrong frozen confirmation protocol")
    comparison = audit["comparisons"][SOURCES[0][1]]
    deltas = [row["delta"]["final"] for row in comparison["per_seed"]]
    interval = comparison["paired_delta_95_percentile_interval"]
    return {"supports_current_cohort_promotion": len(deltas) == 3 and min(deltas) > 0 and interval[0] > 0,
            "candidate_exact_mean_final": audit["exact_mean_final"],
            "paired_mean_delta": comparison["mean_delta"], "paired_delta_95_interval": interval,
            "per_seed_deltas": deltas, "whole_project_blindness_certified": False,
            "ancestor_training_provenance_certified": False, "online_confirmation": False,
            "caveat": "conditional current-cohort confirmation; historical overlap and incomplete ancestor provenance remain"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    plan = make_plan()
    outputs = [ROOT / f"benchmarks/{label}_audit_offset{OFFSET}.json" for _, label, _ in SOURCES]
    outputs += [path for _, label, _ in SOURCES for path in score_paths(ROOT / "benchmarks", label, SEEDS, OFFSET)]
    outputs.append(RESULT_PATH)
    if any(path.exists() for path in outputs):
        raise FileExistsError("confirmation output exists: inspect, never repeat or overwrite")
    if args.register_only:
        with PLAN_PATH.open("x", encoding="utf-8") as stream:
            json.dump(plan, stream, indent=2)
        print(json.dumps({"registered": str(PLAN_PATH), "evaluation_started": False}), flush=True)
        return
    registered = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if registered != plan:
        raise RuntimeError("freeze differs from registered plan")
    lock = ROOT / "artifacts/v242e_confirmation.lock"
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

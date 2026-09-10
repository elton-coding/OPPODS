"""Frozen V257 versus V250; new cohort window, with historical-overlap caveats."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import psutil
from audit_pure_neural_candidate import MODEL_FILES, fingerprint, score_paths
from compare_paired_holdout import compare
from run_pure_neural_long_budget_v250 import data_record, require_report
from run_pure_neural_lr_v237 import ROOT, fingerprints
from run_pure_neural_rms_budget_v242 import require_audit
from run_pure_neural_shared_v257 import require_result
from run_storage_factorial_v254 import verify_bound_inputs, wait_gpu_slot, write_new

SEEDS = [52701, 52702, 52703]
OFFSET = 8000
PROTOCOL = {"split_seed": 1176, "test_offset": OFFSET, "samples": 2000, "noise_seeds": SEEDS}
SOURCES = [
    ("v250_eight_rms_72k", "v250_confirm_v257", "artifacts/pure_neural_v250/eight/steps72000"),
    ("v257_eight_shared_prefix8_rms_72k", "v257_confirm", "artifacts/pure_neural_v257/eight/shared8_steps72000"),
]
INVENTORY = "benchmarks/confirmation_channel_inventory_20260910_1535.json"
PLAN_PATH = ROOT / "benchmarks/v257_confirmation_plan.json"
RESULT_PATH = ROOT / "benchmarks/v257_confirmation_decision.json"
QUEUE_RUNNERS = {
    "run_pure_neural_endtoend_v258.py", "run_pure_neural_rms_global_v259.py",
    "run_pure_neural_receiver_v260.py", "run_pure_neural_rms_hard_v261.py",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def command(label, directory, baseline):
    return [sys.executable, "-u", "scripts/audit_pure_neural_candidate.py", "--submission", directory,
            "--label", label, "--baseline-label", baseline, "--test-offset", str(OFFSET),
            "--samples", "2000", "--noise-seeds", *map(str, SEEDS)]


def decision(audit):
    if audit.get("label") != SOURCES[1][1] or audit.get("protocol") != PROTOCOL:
        raise ValueError("wrong frozen confirmation protocol")
    comparison = audit["comparisons"][SOURCES[0][1]]
    rows = comparison["per_seed"]
    if [row["noise_seed"] for row in rows] != SEEDS:
        raise ValueError("missing, duplicated, or reordered seeds")
    deltas = [row["delta"]["final"] for row in rows]
    interval = comparison["paired_delta_95_percentile_interval"]
    values = [*deltas, *interval, comparison["mean_delta"], audit["exact_mean_final"]]
    if len(interval) != 2 or not all(math.isfinite(v) for v in values) or interval[0] > interval[1]:
        raise ValueError("invalid confirmation statistics")
    return {"supports_current_cohort_promotion": min(deltas) > 0 and interval[0] > 0,
            "candidate_exact_mean_final": audit["exact_mean_final"],
            "paired_mean_delta": comparison["mean_delta"], "paired_delta_95_interval": interval,
            "per_seed_deltas": deltas, "per_seed_metric_deltas": [row["delta"] for row in rows],
            "automatic_promotion": False, "whole_project_blindness_certified": False,
            "ancestor_training_provenance_certified": False, "online_confirmation": False,
            "caveat": "conditional cohort confirmation only; historical overlap and parent provenance gaps remain"}


def checked_window(inventory):
    window = next(row for row in inventory["windows"] if row["test_offset"] == OFFSET)
    if (window["samples"] != 2000 or inventory["unreadable_archives"] != 0
            or any(window[key] for key in ("train1176_overlap", "validation1176_overlap",
                                          "current_offset2000_audit_overlap",
                                          "explicit_split1176_recorded_ids_overlap"))):
        raise ValueError("window has current-protocol reuse or unreadable provenance")
    return window


def make_plan():
    sources, reports, paths = [], [], []
    for source_label, label, directory in SOURCES:
        audit_path = ROOT / f"benchmarks/{source_label}_audit_offset2000.json"
        report = read(audit_path)
        require_audit(report, source_label, 72000)
        (require_result(report["training_report"]) if source_label == SOURCES[1][0]
         else require_report(report["training_report"], 72000))
        if read(ROOT / directory / "training_report.json") != report["training_report"]:
            raise RuntimeError("full training report changed")
        files = fingerprint(ROOT / directory)
        if files != report["files"]:
            raise RuntimeError("model changed since completed source audit")
        caches = score_paths(ROOT / "benchmarks", source_label, [22701, 22702, 22703], 2000)
        for path in caches:
            with np.load(path, allow_pickle=False) as cache:
                if cache["length"].shape != (4000,) or not np.all(cache["length"] == 1152):
                    raise ValueError("source cache must contain all 1152 bits for 4000 UEs")
        paths += [audit_path, *caches, ROOT / directory / "training_report.json"]
        paths += [ROOT / directory / name for name in MODEL_FILES]
        sources.append({"source_audit": str(audit_path.relative_to(ROOT)), "label": label,
                        "directory": directory, "requested_steps": 72000, "files": files})
        reports.append(report)
    selection = reports[1]["comparisons"][SOURCES[0][0]]
    reproduced = compare(*[score_paths(ROOT / "benchmarks", row[0], [22701, 22702, 22703], 2000)
                           for row in SOURCES])
    if selection != reproduced:
        raise RuntimeError("selection statistics do not reproduce from paired caches")
    if (min(row["delta"]["final"] for row in selection["per_seed"]) <= 0
            or selection["paired_delta_95_percentile_interval"][0] <= 0):
        raise ValueError("candidate failed preregistered selection gate")
    for name in ("execution_plan.json", "completed_dependency.json"):
        path = ROOT / "artifacts/pure_neural_v257/eight" / name
        bound = read(path)
        verify_bound_inputs(bound)
        paths += [path, *[ROOT / name for name in bound["input_sha256"]]]
    paths += [ROOT / name for name in (
        "scripts/run_frozen_confirmation_v257.py", "scripts/audit_confirmation_overlap.py",
        "scripts/audit_pure_neural_candidate.py", "scripts/evaluate_submission.py",
        "scripts/compare_paired_holdout.py", "scripts/run_pure_neural_shared_v257.py",
        "scripts/run_pure_neural_long_budget_v250.py", "scripts/run_storage_factorial_v254.py",
        "scripts/run_pure_neural_pair_v253.py", "src/oppods/data.py", INVENTORY)]
    return {"sources": sources, "input_sha256": fingerprints(paths), "dataset": data_record(),
            "protocol": PROTOCOL, "selection_gate_reproduced": True,
            "commands": [command(SOURCES[0][1], SOURCES[0][2], ""),
                         command(SOURCES[1][1], SOURCES[1][2], SOURCES[0][1])],
            "criteria": "all three paired total deltas >0 and channel-bootstrap 95% lower bound >0",
            "scope": "offset8000 unused in current split1176 cohort; offsets4000/6000 not reused",
            "historical_inventory_window": checked_window(read(ROOT / INVENTORY)),
            "resource_policy": "wait for existing V258-V261 runner queue to exit, then acquire GPU audit slot",
            "queue_runners": sorted(QUEUE_RUNNERS), "automatic_promotion": False,
            "whole_project_blindness_certified": False, "ancestor_training_provenance_certified": False,
            "online_confirmation": False}


def is_queue_runner(name, cmd):
    return str(name).lower() == "python.exe" and any(
        str(part).replace("\\", "/").split("/")[-1] in QUEUE_RUNNERS for part in cmd)


def wait_queue(timeout):
    deadline = time.monotonic() + timeout
    while True:
        active = []
        for process in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                if is_queue_runner(process.info["name"], process.info["cmdline"] or []):
                    active.append(process.info["pid"])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if not active:
            wait_gpu_slot(timeout=max(.01, deadline-time.monotonic()))
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"existing queue still active {active}; preserve freeze, do not duplicate")
        time.sleep(min(30., max(.01, deadline-time.monotonic())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--wait-timeout", type=float, default=86400)
    args = parser.parse_args()
    outputs = [ROOT / f"benchmarks/{label}_audit_offset{OFFSET}.json" for _, label, _ in SOURCES]
    outputs += [p for _, label, _ in SOURCES for p in score_paths(ROOT / "benchmarks", label, SEEDS, OFFSET)]
    failure = ROOT / "benchmarks/v257_confirmation_failure.json"
    if any(path.exists() for path in [*outputs, RESULT_PATH, failure]):
        raise FileExistsError("confirmation output exists; never repeat or overwrite")
    plan = make_plan()
    if args.register_only:
        write_new(PLAN_PATH, plan)
        print(json.dumps({"registered": str(PLAN_PATH), "evaluation_started": False}), flush=True)
        return
    registered = read(PLAN_PATH)
    if registered != plan:
        raise RuntimeError("freeze differs from registered plan")
    lock = ROOT / "artifacts/v257_confirmation.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        print(json.dumps({"waiting_for_existing_queue": sorted(QUEUE_RUNNERS), "gpu_evaluation_started": False}), flush=True)
        wait_queue(args.wait_timeout)
        for cmd in plan["commands"]:
            if make_plan() != registered:
                raise RuntimeError("frozen inputs changed before confirmation")
            wait_gpu_slot(timeout=args.wait_timeout)
            subprocess.run(cmd, cwd=ROOT, check=True)
            if make_plan() != registered:
                raise RuntimeError("frozen inputs changed during confirmation")
        audits = [read(path) for path in outputs[:2]]
        for audit, source in zip(audits, plan["sources"], strict=True):
            if audit["protocol"] != PROTOCOL or audit["files"] != source["files"]:
                raise RuntimeError("confirmation source or protocol changed")
            if any(row["short_outputs"] or row["min_output_length"] != 1152
                   or row["max_output_length"] != 1152 or row["scores"] != 4000
                   or row["samples"] != 2000 for row in audit["per_seed"]):
                raise ValueError("incomplete confirmation payload")
        result = decision(audits[1])
        write_new(RESULT_PATH, result)
        print(json.dumps(result), flush=True)
    except Exception as error:
        write_new(failure, {"error": repr(error), "partial_outputs_preserved": True, "implicit_resume_allowed": False})
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()

"""V273: double V269 training budget, fresh144k; no optimizer restart."""
from __future__ import annotations

import math
import os
import subprocess
import sys

from compare_paired_holdout import compare
from run_pure_neural_fairness_v251 import check_initial
from run_pure_neural_long_budget_v250 import require_report
from run_pure_neural_low_tx_untie_v265 import (
    ROOT,
    SEEDS,
    checked_caches,
    data_record,
    decision,
    fingerprint,
    fingerprints,
    read,
    require_audit,
    require_metrics,
    score_paths,
    verify_bound_inputs,
    wait_gpu_slot,
    write_new,
)
from run_shared_lr_v269 import command as baseline_command
from run_shared_lr_v269 import require_result as baseline_guard

CONTROL = "v269_shared8_rms_lr3e5_72k"
CONTROL_DIR = "artifacts/pure_neural_v269/eight/lr3e5_steps72000"
LABEL = "v273_shared8_rms_144k"
BASE = ROOT / "artifacts/pure_neural_v273/eight"
OUTPUT = BASE / "steps144000"
PROBE = ROOT / "artifacts/resource_probe/v273/budget"
PLAN = BASE / "execution_plan.json"
DECISION = ROOT / "benchmarks/v273_selection_decision.json"


def command(probe=False):
    cmd = baseline_command(probe=probe)
    if not probe:
        cmd[cmd.index("--steps")+1] = "144000"
        cmd[cmd.index("--patience")+1] = "144"
    cmd[cmd.index("--output-dir")+1] = str(PROBE if probe else OUTPUT)
    return cmd


def require_result(report, probe=False):
    if probe:
        baseline_guard(report, probe=True)
        return
    require_report({**report, "learning_rate": 1e-5}, 144000)
    # Validate all V269 factors against a budget-normalized VIEW, never change evidence.
    prefix = [row for row in report["history"] if row["step"] <= 72000]
    baseline_guard({**report, "requested_steps": 72000, "history": prefix,
                    "best_step": max(prefix, key=lambda r: r["final"])["step"]})
    if any(not math.isfinite(row[k]) for row in report["history"]
           for k in ("loss", "efficiency", "fairness", "final")):
        raise ValueError("nonfinite validation")
    if report["best_step"] != max(report["history"], key=lambda r: r["final"])["step"]:
        raise ValueError("checkpoint not selected by registered validation")


def check_prefix(report, control):
    rows = [r for r in report["history"] if r["step"] <= 72000]
    if [r["step"] for r in rows] != list(range(0, 72001, 1000)):
        raise ValueError("incomplete72k prefix")
    diffs = {k: max(abs(a[k]-b[k]) for a, b in zip(rows, control["history"], strict=True))
             for k in ("loss", "efficiency", "fairness", "final")}
    if any(x > 1e-5 for x in diffs.values()):
        raise ValueError("original72k trajectory did not reproduce")
    return {"checkpoints": 73, "maximum_difference": diffs,
            "scope": "validation trajectory, not bitwise Adam-state proof"}


def main():
    audit_path = ROOT / f"benchmarks/{LABEL}_audit_offset2000.json"
    probe_path = ROOT / "benchmarks/v273_gpu_probe.json"
    prefix_path = ROOT / "benchmarks/v273_initial_check.json"
    lock = BASE / "execution.lock"
    failure = BASE / "failure.json"
    reserved = [OUTPUT, PROBE, PLAN, DECISION, audit_path, probe_path, prefix_path, lock, failure,
                *score_paths(ROOT / "benchmarks", LABEL, SEEDS, 2000)]
    if any(p.exists() for p in reserved):
        raise FileExistsError("V273 reserved outputs exist; no implicit restart")
    previous_path = ROOT / "artifacts/pure_neural_v272/eight/execution_plan.json"
    previous = read(previous_path)
    verify_bound_inputs(previous)
    control = read(ROOT / f"benchmarks/{CONTROL}_audit_offset2000.json")
    baseline_guard(control["training_report"])
    if control["files"] != fingerprint(ROOT / CONTROL_DIR):
        raise ValueError("control weights changed")
    require_metrics(control, checked_caches([CONTROL], offset=2000, seeds=SEEDS)[CONTROL])
    paths = [ROOT / p for p in previous["input_sha256"]] + [previous_path]
    paths += [ROOT / CONTROL_DIR / name for name in
              ("modelDesign.py", "encoder.pth", "transmitter.pth", "receiver.pth", "training_report.json")]
    paths += [ROOT / f"benchmarks/{CONTROL}_audit_offset2000.json",
              ROOT / "benchmarks/v269_selection_decision.json"]
    paths += score_paths(ROOT / "benchmarks", CONTROL, SEEDS, 2000)
    paths += [ROOT / p for p in (
        "scripts/run_shared_budget_v273.py", "tests/test_shared_budget_v273.py",
        "docs/experiments/shared-budget-v273.md")]
    plan = {"input_sha256": fingerprints(paths), "dataset": data_record(), "command": command(),
            "probe_command": command(True), "control": CONTROL, "label": LABEL,
            "single_factor": "budget72000 to144000; V269 unchanged elsewhere;14.4M draws versus7.2M; not equal compute",
            "bootstrap_seed": 227, "bootstrap_repeats": 2000, "automatic_promotion": False}
    BASE.mkdir(parents=True, exist_ok=True)
    write_new(PLAN, plan)
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        wait_gpu_slot()
        subprocess.run(plan["probe_command"], cwd=ROOT, check=True)
        probe = read(PROBE / "training_report.json")
        require_result(probe, True)
        initial = check_initial(probe, control["training_report"])
        if fingerprint(PROBE)["modelDesign.py"] != control["files"]["modelDesign.py"]:
            raise ValueError("probe architecture changed")
        write_new(probe_path, {"report": probe, "initial_differences": initial})
        verify_bound_inputs(plan)
        subprocess.run(plan["command"], cwd=ROOT, check=True)
        report = read(OUTPUT / "training_report.json")
        require_result(report)
        initial = check_initial(report, control["training_report"])
        if fingerprint(OUTPUT)["modelDesign.py"] != control["files"]["modelDesign.py"]:
            raise ValueError("architecture changed")
        write_new(prefix_path, {"initial_differences": initial,
                                "prefix": check_prefix(report, control["training_report"])})
        verify_bound_inputs(plan)
        if data_record() != plan["dataset"]:
            raise ValueError("dataset changed")
        subprocess.run([sys.executable, "-u", "scripts/audit_pure_neural_candidate.py", "--submission", str(OUTPUT),
                        "--label", LABEL, "--baseline-label", CONTROL], cwd=ROOT, check=True)
        audit = read(audit_path)
        require_audit(audit, LABEL, 144000)
        assert audit["files"] == fingerprint(OUTPUT) and audit["training_report"] == report
        caches = checked_caches([CONTROL, LABEL], offset=2000, seeds=SEEDS)
        require_metrics(audit, caches[LABEL])
        require_metrics(control, caches[CONTROL])
        comp = compare(score_paths(ROOT / "benchmarks", CONTROL, SEEDS, 2000),
                       score_paths(ROOT / "benchmarks", LABEL, SEEDS, 2000))
        assert comp == audit["comparisons"][CONTROL]
        result = {**decision(comp), "exact_mean_final": audit["exact_mean_final"]}
        verify_bound_inputs(plan)
        write_new(DECISION, result)
        print({"mean": result["exact_mean_final"], "eligible": result["eligible_for_new_confirmation"]}, flush=True)
    except Exception as error:
        write_new(failure, {"error": repr(error), "partial_outputs_preserved": True})
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()

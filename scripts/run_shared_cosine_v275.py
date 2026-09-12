"""V275: V273 equal144k budget, late cosine schedule; fresh initialization."""
from __future__ import annotations

import os
import subprocess
import sys

from compare_paired_holdout import compare
from probe_pure_neural_cosine_v275 import EVIDENCE, source_paths
from run_pure_neural_fairness_v251 import check_initial
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
from run_shared_budget_v273 import command as baseline_command
from run_shared_budget_v273 import require_result as baseline_guard
from train_pure_neural_cosine_v275 import require_schedule

CONTROL = "v273_shared8_rms_144k"
CONTROL_DIR = "artifacts/pure_neural_v273/eight/steps144000"
LABEL = "v275_shared8_rms_cosine_144k"
BASE = ROOT / "artifacts/pure_neural_v275/eight"
OUTPUT = BASE / "cosine_steps144000"
PROBE = ROOT / "artifacts/resource_probe/v275/budget"
PLAN = BASE / "execution_plan.json"
DECISION = ROOT / "benchmarks/v275_selection_decision.json"


def command(probe=False):
    cmd = baseline_command(probe=probe)
    cmd[2] = "scripts/train_pure_neural_cosine_v275.py"
    if not probe:
        cmd[cmd.index("--steps")+1] = "144000"
        cmd[cmd.index("--patience")+1] = "144"
    cmd[cmd.index("--output-dir")+1] = str(PROBE if probe else OUTPUT)
    return cmd


def require_result(report, probe=False):
    require_schedule(report.get("learning_rate_schedule"), 2 if probe else 144000)
    if report.get("learning_rate_field_scope") != "initial rate only; actual applied rates bound by learning_rate_schedule":
        raise ValueError("missing schedule disclosure")
    baseline_guard({k: v for k, v in report.items() if k != "learning_rate_schedule"}, probe=probe)


def check_prefix(report, control):
    rows = [r for r in report["history"] if r["step"] <= 72000]
    if [r["step"] for r in rows] != list(range(0, 72001, 1000)):
        raise ValueError("incomplete72k prefix")
    diffs = {k: max(abs(a[k]-b[k]) for a, b in zip(rows, [r for r in control["history"] if r["step"] <= 72000], strict=True))
             for k in ("loss", "efficiency", "fairness", "final")}
    if any(x > 1e-5 for x in diffs.values()):
        raise ValueError("original72k trajectory did not reproduce")
    return {"checkpoints": 73, "maximum_difference": diffs,
            "scope": "validation trajectory, not bitwise Adam-state proof"}


def main():
    audit_path = ROOT / f"benchmarks/{LABEL}_audit_offset2000.json"
    probe_path = ROOT / "benchmarks/v275_gpu_probe.json"
    prefix_path = ROOT / "benchmarks/v275_initial_check.json"
    lock = BASE / "execution.lock"
    failure = BASE / "failure.json"
    reserved = [OUTPUT, PROBE, PLAN, DECISION, audit_path, probe_path, prefix_path, lock, failure,
                *score_paths(ROOT / "benchmarks", LABEL, SEEDS, 2000)]
    if any(p.exists() for p in reserved):
        raise FileExistsError("V275 reserved outputs exist; no implicit restart")
    previous_path = ROOT / "artifacts/pure_neural_v273/eight/execution_plan.json"
    previous = read(previous_path)
    verify_bound_inputs(previous)
    cpu = read(ROOT / EVIDENCE)
    if cpu.get("passed") is not True or cpu.get("input_sha256") != fingerprints(source_paths()):
        raise ValueError("stale CPU proof")
    require_schedule(cpu.get("actual_schedule"), 144000)
    control = read(ROOT / f"benchmarks/{CONTROL}_audit_offset2000.json")
    baseline_guard(control["training_report"])
    if control["files"] != fingerprint(ROOT / CONTROL_DIR):
        raise ValueError("control weights changed")
    require_metrics(control, checked_caches([CONTROL], offset=2000, seeds=SEEDS)[CONTROL])
    paths = [ROOT / p for p in previous["input_sha256"]] + [previous_path]
    paths += [ROOT / CONTROL_DIR / name for name in
              ("modelDesign.py", "encoder.pth", "transmitter.pth", "receiver.pth", "training_report.json")]
    paths += [ROOT / f"benchmarks/{CONTROL}_audit_offset2000.json",
              ROOT / "benchmarks/v273_selection_decision.json"]
    paths += score_paths(ROOT / "benchmarks", CONTROL, SEEDS, 2000)
    paths += source_paths() + [ROOT / EVIDENCE]
    paths += [ROOT / p for p in (
        "scripts/run_shared_cosine_v275.py", "tests/test_shared_cosine_v275.py",
        "docs/experiments/shared-cosine-v275.md")]
    plan = {"input_sha256": fingerprints(paths), "dataset": data_record(), "command": command(),
            "probe_command": command(True), "control": CONTROL, "label": LABEL,
            "single_factor": "constant72k3e-5 then cosine72k to1e-5; V273 unchanged elsewhere; equal14.4M draws",
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

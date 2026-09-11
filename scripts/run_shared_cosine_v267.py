"""V267: V257 architecture with the unchanged V255 late-cosine schedule."""
from __future__ import annotations

import math
import os
import subprocess
import sys

from compare_paired_holdout import compare
from probe_pure_neural_cosine_v255 import EVIDENCE, source_paths
from run_pure_neural_fairness_v251 import check_initial
from run_pure_neural_long_budget_v250 import check_prefix
from run_pure_neural_low_tx_untie_v265 import (
    CONTROL,
    CONTROL_DIR,
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
from run_pure_neural_shared_v257 import command as baseline_command
from run_pure_neural_shared_v257 import require_result as baseline_guard
from train_pure_neural_cosine_v255 import require_schedule

LABEL = "v267_shared8_rms_cosine_72k"
BASE = ROOT / "artifacts/pure_neural_v267/eight"
OUTPUT = BASE / "cosine_steps72000"
PROBE = ROOT / "artifacts/resource_probe/v267/shared_cosine"
PLAN = BASE / "execution_plan.json"
DECISION = ROOT / "benchmarks/v267_selection_decision.json"


def command(probe=False):
    cmd = baseline_command(probe=probe)
    cmd[2] = "scripts/train_pure_neural_cosine_v255.py"
    cmd[cmd.index("--output-dir")+1] = str(PROBE if probe else OUTPUT)
    return cmd


def require_result(report, probe=False):
    require_schedule(report.get("learning_rate_schedule"), 2 if probe else 72000)
    if report.get("learning_rate_field_scope") != "initial rate only; actual applied rates bound by learning_rate_schedule":
        raise ValueError("missing actual schedule disclosure")
    baseline_guard({k: v for k, v in report.items() if k != "learning_rate_schedule"}, probe=probe)
    if any(k in report for k in ("encoder_learning_rate", "microbatch_size")):
        raise ValueError("extra training factor")
    if any(not math.isfinite(row[k]) for row in report["history"]
           for k in ("loss", "efficiency", "fairness", "final")):
        raise ValueError("nonfinite validation")
    if report.get("best_step") not in [r["step"] for r in report["history"]]:
        raise ValueError("best checkpoint not in trajectory")


def main():
    audit_path = ROOT / f"benchmarks/{LABEL}_audit_offset2000.json"
    probe_path = ROOT / "benchmarks/v267_gpu_probe.json"
    prefix_path = ROOT / "benchmarks/v267_prefix_check.json"
    lock = BASE / "execution.lock"
    failure = BASE / "failure.json"
    reserved = [OUTPUT, PROBE, PLAN, DECISION, audit_path, probe_path, prefix_path, lock, failure,
                *score_paths(ROOT / "benchmarks", LABEL, SEEDS, 2000)]
    if any(p.exists() for p in reserved):
        raise FileExistsError("V267 reserved outputs exist; no implicit restart")
    previous_path = ROOT / "artifacts/pure_neural_v265/eight/execution_plan.json"
    previous = read(previous_path)
    verify_bound_inputs(previous)
    cpu = read(ROOT / EVIDENCE)
    if (cpu.get("passed") is not True or cpu.get("updates_checked") != 72000
            or cpu.get("input_sha256") != fingerprints(source_paths())):
        raise ValueError("schedule proof stale")
    require_schedule(cpu.get("actual_schedule"), 72000)
    control = read(ROOT / f"benchmarks/{CONTROL}_audit_offset2000.json")
    baseline_guard(control["training_report"])
    if control["files"] != fingerprint(ROOT / CONTROL_DIR):
        raise ValueError("control weights changed")
    require_metrics(control, checked_caches([CONTROL], offset=2000, seeds=SEEDS)[CONTROL])
    paths = [ROOT / p for p in previous["input_sha256"]] + [previous_path, ROOT / EVIDENCE]
    paths += source_paths() + [ROOT / p for p in (
        "scripts/run_shared_cosine_v267.py", "tests/test_shared_cosine_v267.py",
        "docs/experiments/shared-cosine-v267.md")]
    plan = {"input_sha256": fingerprints(paths), "dataset": data_record(), "command": command(),
            "probe_command": command(True), "control": CONTROL, "label": LABEL,
            "single_factor": "constant36k then V255 cosine36k; V257 unchanged elsewhere",
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
        prefix = check_prefix(report, {"history": [r for r in control["training_report"]["history"] if r["step"] <= 36000]})
        if not prefix["matched"]:
            raise ValueError("constant prefix differs from V257")
        if fingerprint(OUTPUT)["modelDesign.py"] != control["files"]["modelDesign.py"]:
            raise ValueError("architecture changed")
        write_new(prefix_path, prefix)
        verify_bound_inputs(plan)
        if data_record() != plan["dataset"]:
            raise ValueError("dataset changed")
        subprocess.run([sys.executable, "-u", "scripts/audit_pure_neural_candidate.py", "--submission", str(OUTPUT),
                        "--label", LABEL, "--baseline-label", CONTROL], cwd=ROOT, check=True)
        audit = read(audit_path)
        require_audit(audit, LABEL, 72000)
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

"""V271: only increase V269 RMS constant learning rate3e-5 to3.5e-5; fresh full72k."""
from __future__ import annotations

import math
import os
import subprocess
import sys

from compare_paired_holdout import compare
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
from run_shared_lr_v269 import command as baseline_command
from run_shared_lr_v269 import require_result as baseline_guard

CONTROL = "v269_shared8_rms_lr3e5_72k"
CONTROL_DIR = "artifacts/pure_neural_v269/eight/lr3e5_steps72000"
LABEL = "v271_shared8_rms_lr35e6_72k"
BASE = ROOT / "artifacts/pure_neural_v271/eight"
OUTPUT = BASE / "lr35e6_steps72000"
PROBE = ROOT / "artifacts/resource_probe/v271/shared_lr"
PLAN = BASE / "execution_plan.json"
DECISION = ROOT / "benchmarks/v271_selection_decision.json"


def command(probe=False):
    cmd = baseline_command(probe=probe)
    cmd[cmd.index("--learning-rate")+1] = "3.5e-5"
    cmd[cmd.index("--output-dir")+1] = str(PROBE if probe else OUTPUT)
    return cmd


def require_result(report, probe=False):
    if report.get("learning_rate") != 3.5e-5 or "learning_rate_schedule" in report:
        raise ValueError("V271 requires constant3.5e-5 without schedule")
    baseline_guard({**report, "learning_rate": 3e-5}, probe=probe)
    if any(k in report for k in ("encoder_learning_rate", "microbatch_size")):
        raise ValueError("extra training factor")
    if any(not math.isfinite(row[k]) for row in report["history"]
           for k in ("loss", "efficiency", "fairness", "final")):
        raise ValueError("nonfinite validation")
    if report.get("best_step") not in [r["step"] for r in report["history"]]:
        raise ValueError("best checkpoint not in trajectory")


def main():
    audit_path = ROOT / f"benchmarks/{LABEL}_audit_offset2000.json"
    probe_path = ROOT / "benchmarks/v271_gpu_probe.json"
    prefix_path = ROOT / "benchmarks/v271_initial_check.json"
    lock = BASE / "execution.lock"
    failure = BASE / "failure.json"
    reserved = [OUTPUT, PROBE, PLAN, DECISION, audit_path, probe_path, prefix_path, lock, failure,
                *score_paths(ROOT / "benchmarks", LABEL, SEEDS, 2000)]
    if any(p.exists() for p in reserved):
        raise FileExistsError("V271 reserved outputs exist; no implicit restart")
    previous_path = ROOT / "artifacts/pure_neural_v270/eight/execution_plan.json"
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
        "scripts/run_shared_lr_v271.py", "tests/test_shared_lr_v271.py",
        "docs/experiments/shared-lr-v271.md")]
    plan = {"input_sha256": fingerprints(paths), "dataset": data_record(), "command": command(),
            "probe_command": command(True), "control": CONTROL, "label": LABEL,
            "single_factor": "constant learning rate3e-5 to3.5e-5; V269 unchanged elsewhere",
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
        write_new(prefix_path, {"initial_differences": initial})
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

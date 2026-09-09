"""V256: RMS temperature .5 -> .25 only, original constant-LR full72k control."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from audit_pure_neural_candidate import fingerprint, score_paths
from run_pure_neural_cosine_v255 import require_result as require_cosine_result
from run_pure_neural_fairness_v251 import check_initial
from run_pure_neural_long_budget_v250 import command as control_command
from run_pure_neural_long_budget_v250 import data_record, require_report
from run_pure_neural_lr_v237 import ROOT, fingerprints
from run_pure_neural_rms_budget_v242 import require_audit
from run_storage_factorial_v254 import verify_bound_inputs, wait_gpu_slot, write_new

CONTROL = "v250_eight_rms_72k"
DEPENDENCY = "v255_eight_rms_cosine_72k"
LABEL = "v256_eight_rms_temperature025_72k"
OUTPUT = "artifacts/pure_neural_v256/eight/temperature025_steps72000"
PROBE = "artifacts/resource_probe/v256/eight_temperature025"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def command(*, probe=False):
    result = control_command()
    for flag, value in (("--score-temperature", "0.25"), ("--output-dir", PROBE if probe else OUTPUT)):
        result[result.index(flag) + 1] = value
    if probe:
        for flag in ("--steps", "--validate-every"):
            result[result.index(flag) + 1] = "2"
    return result


def require_result(report, *, probe=False):
    if report.get("score_temperature") != .25 or "learning_rate_schedule" in report:
        raise ValueError("V256 must use temperature.25 and constant LR, not a cosine combination")
    if probe:
        expected = {"requested_steps": 2, "batch_size": 100, "seed": 15240, "learning_rate": 1e-5,
                    "loss_kind": "rms_score", "score_bce_weight": .05, "score_fairness_weight": .3,
                    "quantile_bandwidth": .025, "train_components": ["transmitter", "receiver"],
                    "validation_samples": 2000, "baseline_expert_map": [0, 0, 1, 1, 1, 1, 1, 1],
                    "gpu_memory_fraction": .4}
        if (any(report.get(k) != v for k, v in expected.items())
                or [r["step"] for r in report.get("history", [])] != [0, 2]
                or report.get("gpu_peak_allocated_bytes", 0) <= 0):
            raise ValueError("V256 incomplete/mismatched initial GPU probe")
    else:
        # Validate all unchanged fields without modifying the actual report.
        require_report({**report, "score_temperature": .5}, 72000)
    if report.get("trainable_parameters") != 189551584 or report.get("variable_payload") is not False:
        raise ValueError("V256 must keep original eight-expert full payload architecture")


def wait_dependency(timeout):
    path = ROOT / f"benchmarks/{DEPENDENCY}_audit_offset2000.json"
    deadline = time.monotonic() + timeout
    print(json.dumps({"waiting_for_full_audit": DEPENDENCY, "steps": 72000}), flush=True)
    while True:
        if path.exists():
            try:
                audit = read(path)
            except json.JSONDecodeError:
                pass
            else:
                require_audit(audit, DEPENDENCY, 72000)
                require_cosine_result(audit["training_report"])
                if fingerprint(ROOT / "artifacts/pure_neural_v255/eight/cosine_steps72000") != audit["files"]:
                    raise RuntimeError("completed dependency files changed")
                return audit
        if time.monotonic() >= deadline:
            raise TimeoutError("V255 incomplete; inspect its real process, do not duplicate or resume")
        time.sleep(min(30., max(.01, deadline - time.monotonic())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-timeout", type=float, default=86400)
    args = parser.parse_args()
    output, probe = ROOT / OUTPUT, ROOT / PROBE
    reserved = [output, probe, output.parent / "execution_plan.json", output.parent / "execution.lock",
                output.parent / "failure.json", ROOT / "benchmarks/v256_gpu_training_probe.json",
                ROOT / "benchmarks/v256_initial_validation_check.json",
                ROOT / f"benchmarks/{LABEL}_audit_offset2000.json"]
    reserved += score_paths(ROOT / "benchmarks", LABEL, [22701, 22702, 22703], 2000)
    if any(path.exists() for path in reserved):
        raise FileExistsError("V256 output exists; refusing implicit restart/resume/overwrite")
    control_path = ROOT / f"benchmarks/{CONTROL}_audit_offset2000.json"
    control = read(control_path)
    require_audit(control, CONTROL, 72000)
    require_report(control["training_report"], 72000)
    if fingerprint(ROOT / "artifacts/pure_neural_v250/eight/steps72000") != control["files"]:
        raise RuntimeError("frozen V250 control changed")
    paths = [control_path]
    for name in ("artifacts/pure_neural_v250/eight/execution_plan.json",
                 "artifacts/pure_neural_v255/eight/execution_plan.json"):
        path = ROOT / name
        bound = read(path)
        verify_bound_inputs(bound)
        paths += [path, *[ROOT / p for p in bound["input_sha256"]]]
    paths += [ROOT / p for p in ("scripts/run_pure_neural_rms_temperature_v256.py",
              "scripts/run_pure_neural_cosine_v255.py", "scripts/run_pure_neural_long_budget_v250.py",
              "scripts/run_pure_neural_fairness_v251.py", "scripts/run_storage_factorial_v254.py",
              "scripts/audit_pure_neural_candidate.py", "scripts/evaluate_submission.py",
              "scripts/compare_paired_holdout.py")]
    plan = {"label": LABEL, "control": CONTROL, "dependency": DEPENDENCY,
            "command": command(), "probe_command": command(probe=True),
            "input_sha256": fingerprints(paths), "dataset": data_record(), "wait_timeout": args.wait_timeout,
            "single_factor": "RMS proxy temperature0.5 to0.25; unchanged raw BCE0.05 and constant Adam1e-5",
            "train_channel_draws": 7200000, "initialization": "original V227; fresh Adam, not V255/probe weights",
            "resource_gate": "unchanged V250 architecture/batch; original entry two-step/full2000-validation probe",
            "temperature_selected_before_new_training_scores": .25, "automatic_promotion": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / "execution.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        write_new(output.parent / "execution_plan.json", plan)
        print(json.dumps({"plan": str(output.parent / "execution_plan.json"), "frozen_inputs": len(plan["input_sha256"])}), flush=True)
        dependency = wait_dependency(args.wait_timeout)
        if max(control["exact_mean_final"], dependency["exact_mean_final"]) >= 69:
            print("Completed local audit reached69; defer new training for frozen confirmation.", flush=True)
            return
        verify_bound_inputs(plan)
        if data_record() != plan["dataset"]:
            raise RuntimeError("data changed while waiting")
        wait_gpu_slot()
        subprocess.run(plan["probe_command"], cwd=ROOT, check=True)
        probe_report = read(probe / "training_report.json")
        require_result(probe_report, probe=True)
        initial = check_initial(probe_report, control["training_report"])
        write_new(ROOT / "benchmarks/v256_gpu_training_probe.json", {"purpose": "initialization/runtime only, not score",
                  "report": probe_report, "initial_validation_differences": initial})
        verify_bound_inputs(plan)
        wait_gpu_slot()
        subprocess.run(plan["command"], cwd=ROOT, check=True)
        report = read(output / "training_report.json")
        require_result(report)
        initial = check_initial(report, control["training_report"])
        verify_bound_inputs(plan)
        if data_record() != plan["dataset"]:
            raise RuntimeError("data changed during formal training")
        write_new(ROOT / "benchmarks/v256_initial_validation_check.json", {"maximum_absolute_differences": initial, "matched": True})
        wait_gpu_slot()
        subprocess.run([sys.executable, "-u", "scripts/audit_pure_neural_candidate.py", "--submission", OUTPUT,
                        "--label", LABEL, "--baseline-label", CONTROL], cwd=ROOT, check=True)
        verify_bound_inputs(plan)
    except Exception as error:
        write_new(output.parent / "failure.json", {"error": repr(error), "partial_outputs_preserved": True,
                                                   "implicit_resume_allowed": False})
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()

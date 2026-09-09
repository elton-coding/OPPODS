"""Wait for V253 completion, then independently train the registered V255 late-cosine arm."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import torch
from audit_pure_neural_candidate import fingerprint
from probe_pure_neural_cosine_v255 import EVIDENCE, source_paths
from run_pure_neural_fairness_v251 import check_initial
from run_pure_neural_long_budget_v250 import check_prefix, data_record, require_report
from run_pure_neural_long_budget_v250 import command as control_command
from run_pure_neural_lr_v237 import ROOT, fingerprints
from run_pure_neural_rms_budget_v242 import require_audit
from run_pure_neural_routing_v252 import require_result as require_routing_result
from run_storage_factorial_v254 import verify_bound_inputs, wait_gpu_slot, write_new
from train_pure_neural_cosine_v255 import require_schedule, schedule_definition

CONTROL = "v250_eight_rms_72k"
LABEL = "v255_eight_rms_cosine_72k"
DEPENDENCY = "v253_pair_routing_rms_36k"
OUTPUT = "artifacts/pure_neural_v255/eight/cosine_steps72000"
PROBE = "artifacts/resource_probe/v255/eight_cosine"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def command(*, probe=False):
    result = control_command()
    result[2] = "scripts/train_pure_neural_cosine_v255.py"
    result[result.index("--output-dir") + 1] = PROBE if probe else OUTPUT
    if probe:
        for flag in ("--steps", "--validate-every"):
            result[result.index(flag) + 1] = "2"
    return result


def require_result(report, *, probe=False):
    steps = 2 if probe else 72000
    if probe:
        expected = {"requested_steps": 2, "batch_size": 100, "seed": 15240, "learning_rate": 1e-5,
                    "loss_kind": "rms_score", "score_temperature": .5, "score_bce_weight": .05,
                    "score_fairness_weight": .3, "quantile_bandwidth": .025,
                    "train_components": ["transmitter", "receiver"], "validation_samples": 2000,
                    "baseline_expert_map": [0, 0, 1, 1, 1, 1, 1, 1], "gpu_memory_fraction": .4}
        if (any(report.get(k) != v for k, v in expected.items())
                or [r["step"] for r in report.get("history", [])] != [0, 2]
                or report.get("gpu_peak_allocated_bytes", 0) <= 0):
            raise ValueError("V255 incomplete or mismatched two-step GPU probe")
    else:
        require_report(report, 72000)
    if report.get("trainable_parameters") != 189551584 or report.get("variable_payload") is not False:
        raise ValueError("V255 must retain original eight-expert full payload architecture")
    require_schedule(report.get("learning_rate_schedule"), steps)
    if report.get("learning_rate_field_scope") != "initial rate only; actual applied rates bound by learning_rate_schedule":
        raise ValueError("report must distinguish initial LR from actual schedule")


def wait_dependency(timeout):
    path = ROOT / f"benchmarks/{DEPENDENCY}_audit_offset2000.json"
    deadline = time.monotonic() + timeout
    print(json.dumps({"waiting_for_full_audit": DEPENDENCY, "steps": 36000}), flush=True)
    while True:
        if path.exists():
            try:
                audit = read(path)
            except json.JSONDecodeError:
                pass
            else:
                require_audit(audit, DEPENDENCY, 36000)
                require_routing_result(audit["training_report"])
                if fingerprint(ROOT / "artifacts/pure_neural_v253/eight/pair_steps36000") != audit["files"]:
                    raise ValueError("completed V253 files differ from audit")
                return audit
        if time.monotonic() >= deadline:
            raise TimeoutError("V253 complete audit unavailable; inspect existing process, do not duplicate")
        time.sleep(min(30., max(.01, deadline - time.monotonic())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-timeout", type=float, default=43200)
    args = parser.parse_args()
    output, probe = ROOT / OUTPUT, ROOT / PROBE
    reserved = [output, probe, output.parent / "execution_plan.json", output.parent / "execution.lock",
                output.parent / "failure.json", ROOT / "benchmarks/v255_gpu_training_probe.json",
                ROOT / "benchmarks/v255_cosine_prefix_check.json",
                ROOT / f"benchmarks/{LABEL}_audit_offset2000.json"]
    reserved += [ROOT / f"benchmarks/{LABEL}_holdout1176_offset2000_noise{s}.npz" for s in (22701, 22702, 22703)]
    if any(p.exists() for p in reserved):
        raise FileExistsError("V255 output exists; refuse implicit resume/restart")
    cpu = read(ROOT / EVIDENCE)
    if (cpu.get("passed") is not True or cpu.get("updates_checked") != 72000
            or cpu.get("constant_prefix_parameter_and_adam_state_exact_updates") != 36000
            or cpu.get("manual_schedule_parameter_and_adam_state_exact_updates") != 72000
            or cpu.get("global_and_local_rng_unchanged") is not True or cpu.get("torch_version") != torch.__version__
            or cpu.get("input_sha256") != fingerprints(source_paths())):
        raise ValueError("CPU full schedule proof missing or stale")
    require_schedule(cpu["actual_schedule"], 72000)
    control_path = ROOT / f"benchmarks/{CONTROL}_audit_offset2000.json"
    control = read(control_path)
    require_audit(control, CONTROL, 72000)
    require_report(control["training_report"], 72000)
    control_dir = ROOT / "artifacts/pure_neural_v250/eight/steps72000"
    if fingerprint(control_dir) != control["files"]:
        raise RuntimeError("registered V250 control changed")
    paths = source_paths() + [control_path, ROOT / EVIDENCE]
    for name in ("artifacts/pure_neural_v250/eight/execution_plan.json",
                 "artifacts/pure_neural_v253/eight/execution_plan.json",
                 "artifacts/pure_neural_v254/storage/execution_plan.json"):
        path = ROOT / name
        bound = read(path)
        verify_bound_inputs(bound)
        paths += [path, *[ROOT / p for p in bound["input_sha256"]]]
    paths += [ROOT / p for p in ("scripts/run_pure_neural_cosine_v255.py",
              "scripts/run_pure_neural_long_budget_v250.py", "scripts/run_pure_neural_fairness_v251.py",
              "scripts/run_pure_neural_routing_v252.py", "scripts/run_storage_factorial_v254.py",
              "scripts/audit_pure_neural_candidate.py", "scripts/evaluate_submission.py",
              "scripts/compare_paired_holdout.py")]
    plan = {"label": LABEL, "control": CONTROL, "dependency": DEPENDENCY,
            "command": command(), "probe_command": command(probe=True), "schedule": schedule_definition(),
            "input_sha256": fingerprints(paths), "dataset": data_record(),
            "single_factor": "constant36k then cosine36k from1e-5 to1e-6; unchanged Adam state",
            "train_channel_draws": 7200000, "initialization": "original V227; fresh Adam, not V250/probe checkpoint",
            "full_prefix_validation_points_required": 37, "wait_timeout": args.wait_timeout}
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / "execution.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        write_new(output.parent / "execution_plan.json", plan)
        print(json.dumps({"plan": str(output.parent / "execution_plan.json"), "frozen_inputs": len(plan["input_sha256"])}), flush=True)
        dependency = wait_dependency(args.wait_timeout)
        if max(control["exact_mean_final"], dependency["exact_mean_final"]) >= 69:
            print("A completed local audit reached69; defer new training for frozen confirmation.", flush=True)
            return
        verify_bound_inputs(plan)
        if data_record() != plan["dataset"]:
            raise RuntimeError("registered dataset changed while waiting")
        wait_gpu_slot()
        subprocess.run(plan["probe_command"], cwd=ROOT, check=True)
        probe_report = read(probe / "training_report.json")
        require_result(probe_report, probe=True)
        initial = check_initial(probe_report, control["training_report"])
        write_new(ROOT / "benchmarks/v255_gpu_training_probe.json",
                  {"purpose": "initialization/runtime only, not score", "report": probe_report,
                   "initial_validation_differences": initial})
        verify_bound_inputs(plan)
        wait_gpu_slot()
        subprocess.run(plan["command"], cwd=ROOT, check=True)
        report = read(output / "training_report.json")
        require_result(report)
        prefix = check_prefix(report, {"history": [r for r in control["training_report"]["history"] if r["step"] <= 36000]})
        write_new(ROOT / "benchmarks/v255_cosine_prefix_check.json", {"control": CONTROL, **prefix})
        verify_bound_inputs(plan)
        if not prefix["matched"] or data_record() != plan["dataset"]:
            raise RuntimeError("V255 prefix or frozen dataset mismatch; not a controlled schedule comparison")
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

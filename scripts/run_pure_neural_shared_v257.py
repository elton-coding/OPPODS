"""V257 same-parent prefix tying: wait for complete V254 storage work, then full72k."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import run_storage_factorial_v254 as storage
from audit_pure_neural_candidate import fingerprint, score_paths
from probe_pure_neural_shared_v257 import DESIGN, EVIDENCE, MAPPING, source_paths
from run_pure_neural_fairness_v251 import check_initial
from run_pure_neural_long_budget_v250 import command as control_command
from run_pure_neural_long_budget_v250 import data_record, require_report
from run_pure_neural_lr_v237 import ROOT, fingerprints
from run_pure_neural_rms_budget_v242 import require_audit
from run_storage_factorial_v254 import verify_bound_inputs, wait_gpu_slot, write_new

CONTROL = "v250_eight_rms_72k"
LABEL = "v257_eight_shared_prefix8_rms_72k"
OUTPUT = "artifacts/pure_neural_v257/eight/shared8_steps72000"
PROBE = "artifacts/resource_probe/v257/eight_shared8"
MEMORY_PROOF = "benchmarks/v257_full_state_gpu_probe.json"
MEMORY_PLAN = "artifacts/resource_probe/v257/full_state/execution_plan.json"
DEPENDENCY_DECISION = ROOT / "artifacts/pure_neural_v254/storage/decision.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def command(*, probe=False):
    result = control_command()
    for flag, value in (("--model-design", DESIGN), ("--output-dir", PROBE if probe else OUTPUT)):
        result[result.index(flag) + 1] = value
    if probe:
        for flag in ("--steps", "--validate-every"):
            result[result.index(flag) + 1] = "2"
    return result


def require_result(report, *, probe=False):
    expected = {"trainable_parameters": 72744448, "variable_payload": False, "stage": "calibrate",
                "tail_fraction": .1, "focus_prob": 0., "train_snr_interval": [-20., 20.],
                "validation_match_train_snr": False, "shared_frontend": False}
    if any(report.get(key) != value for key, value in expected.items()) or "learning_rate_schedule" in report:
        raise ValueError("V257 needs shared-prefix parameters/full payload/original sampling/constant LR")
    if probe:
        probe_fields = {"requested_steps": 2, "batch_size": 100, "seed": 15240, "learning_rate": 1e-5,
                        "loss_kind": "rms_score", "score_temperature": .5, "score_bce_weight": .05,
                        "score_fairness_weight": .3, "quantile_bandwidth": .025,
                        "train_components": ["transmitter", "receiver"], "validation_samples": 2000,
                        "baseline_expert_map": MAPPING, "gpu_memory_fraction": .4}
        if (any(report.get(key) != value for key, value in probe_fields.items())
                or [row["step"] for row in report.get("history", [])] != [0, 2]
                or report.get("gpu_peak_allocated_bytes", 0) <= 0):
            raise ValueError("V257 incomplete/mismatched GPU initial probe")
    else:
        require_report(report, 72000)


def require_memory(record):
    expected = {"passed": True, "batch_size": 100, "trainable_parameters": 72744448,
                "route_counts": [13]*4+[12]*4, "weights_saved": False, "gpu_memory_fraction": .4,
                "shared_parameter_alias_checks_after_cuda": 1024}
    if (any(record.get(key) != value for key, value in expected.items())
            or [row.get("step") for row in record.get("steps", [])] != [1, 2]
            or any(row.get("adam_parameter_states") != 528 or row.get("gradient_tensors") != 528
                   for row in record.get("steps", []))
            or record.get("gpu_peak_allocated_bytes", 0) <= 0):
        raise ValueError("V257 GPU proof must cover all routes, tied parameters and full Adam state")
    verify_bound_inputs(record)


def completed_storage(decision):
    """Resource dependency only; never adopt storage scores as V257 evidence."""
    verify_bound_inputs(read(storage.BASE / "execution_plan.json"))
    verify_bound_inputs(read(storage.BASE / "completed_source_A.json"))
    source = {"C": storage.verify_source("eight"), "A": storage.verify_source("sixteen")}
    storage.verify_conversion("eight", source["C"])
    storage.verify_conversion("sixteen", source["A"])
    audits = {group: read(storage.audit_path(group)) for group in storage.LABELS}
    result = read(storage.FACTORIAL)
    paths = [DEPENDENCY_DECISION, storage.FACTORIAL, storage.BASE / "completed_source_A.json"]
    for group, audit in audits.items():
        paths += [storage.audit_path(group), *storage.checked_scores(group, audit)]
        if fingerprint(storage.DIRECTORIES[group]) != audit["files"]:
            raise RuntimeError("completed dependency model changed")
        paths += [storage.DIRECTORIES[group] / name for name in (*storage.MODEL_FILES, "training_report.json")]
    actual_means = {group: audit["exact_mean_final"] for group, audit in audits.items()}
    if (result.get("exact_evaluator_mean_final") != actual_means
            or decision.get("exact_mean_final") != actual_means
            or decision.get("automatic_promotion") is not False):
        raise ValueError("V254 storage completion must match all four complete audited arms")
    for group in ("A", "B", "AB"):
        path = storage.BASE / "packages" / f"{group}_{storage.LABELS[group]}.zip"
        if storage.verify_archive(path, audits[group]["files"]) != result["packages"][group]:
            raise RuntimeError("completed dependency package evidence changed")
        paths.append(path)
    return {"input_sha256": fingerprints(paths), "exact_mean_final": actual_means,
            "resource_dependency_only": True, "v257_control": CONTROL}


def wait_dependency(timeout):
    deadline = time.monotonic()+timeout
    print(json.dumps({"waiting_for_storage_decision": str(DEPENDENCY_DECISION)}), flush=True)
    while True:
        if DEPENDENCY_DECISION.exists():
            try:
                decision = read(DEPENDENCY_DECISION)
            except json.JSONDecodeError:
                pass
            else:
                return completed_storage(decision)
        elif (DEPENDENCY_DECISION.parent / "failure.json").exists():
            raise RuntimeError("V254 storage reported failure; inspect original job, do not bypass it")
        if time.monotonic() >= deadline:
            raise TimeoutError("V254 storage incomplete; preserve outputs and inspect the original jobs")
        time.sleep(min(30., max(.01, deadline-time.monotonic())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-timeout", type=float, default=86400)
    args = parser.parse_args()
    output, probe = ROOT / OUTPUT, ROOT / PROBE
    reserved = [output, probe, output.parent / "execution_plan.json", output.parent / "execution.lock",
                output.parent / "failure.json", output.parent / "completed_dependency.json",
                ROOT / MEMORY_PROOF, ROOT / MEMORY_PLAN, ROOT / "benchmarks/v257_gpu_training_probe.json",
                ROOT / "benchmarks/v257_initial_validation_check.json",
                ROOT / f"benchmarks/{LABEL}_audit_offset2000.json"]
    reserved += score_paths(ROOT / "benchmarks", LABEL, [22701, 22702, 22703], 2000)
    if any(path.exists() for path in reserved):
        raise FileExistsError("V257 output exists; refusing implicit restart/resume/overwrite")
    cpu = read(ROOT / EVIDENCE)
    if (cpu.get("passed") is not True or cpu.get("trainable_parameters") != 72744448
            or cpu.get("input_sha256") != fingerprints(source_paths())):
        raise ValueError("V257 CPU proof missing, mismatched or stale")
    control_path = ROOT / f"benchmarks/{CONTROL}_audit_offset2000.json"
    control = read(control_path)
    require_audit(control, CONTROL, 72000)
    require_report(control["training_report"], 72000)
    if fingerprint(ROOT / "artifacts/pure_neural_v250/eight/steps72000") != control["files"]:
        raise RuntimeError("frozen V250 control changed")
    original_path = ROOT / "artifacts/pure_neural_v256/eight/execution_plan.json"
    original = read(original_path)
    verify_bound_inputs(original)
    paths = source_paths() + [control_path, original_path, ROOT / EVIDENCE]
    paths += [ROOT / name for name in original["input_sha256"]]
    paths += [ROOT / name for name in ("scripts/run_pure_neural_shared_v257.py",
              "scripts/probe_pure_neural_shared_memory_v257.py", "scripts/run_pure_neural_pair_v253.py",
              "scripts/run_storage_factorial_v254.py", "scripts/run_pure_neural_long_budget_v250.py",
              "scripts/run_pure_neural_fairness_v251.py", "scripts/audit_pure_neural_candidate.py",
              "scripts/evaluate_submission.py", "scripts/compare_paired_holdout.py")]
    plan = {"label": LABEL, "control": CONTROL, "command": command(), "probe_command": command(probe=True),
            "input_sha256": fingerprints(paths), "dataset": data_record(), "wait_timeout": args.wait_timeout,
            "single_factor": "same-parent eight-expert prefix sharing depth0 to8; private two-block suffix/head",
            "train_channel_draws": 7200000, "initialization": "original V227; fresh Adam, no probe/V250 reuse",
            "dependency": str(DEPENDENCY_DECISION), "dependency_scope": "complete V254 storage before taking its slot",
            "trainable_parameters": 72744448, "trainable_parameter_tensors": 528,
            "shared_prefix_blocks": 8, "parent_map": MAPPING, "automatic_promotion": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / "execution.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        write_new(output.parent / "execution_plan.json", plan)
        print(json.dumps({"plan": str(output.parent / "execution_plan.json"),
                          "frozen_inputs": len(plan["input_sha256"])}), flush=True)
        dependency = wait_dependency(args.wait_timeout)
        write_new(output.parent / "completed_dependency.json", dependency)
        if max(dependency["exact_mean_final"].values()) >= 69:
            print("Dependency audit reached local69; defer new training for confirmation/compliance review.", flush=True)
            return
        verify_bound_inputs(plan)
        if data_record() != plan["dataset"]:
            raise RuntimeError("data changed while waiting")
        wait_gpu_slot()
        subprocess.run([sys.executable, "-u", "scripts/probe_pure_neural_shared_memory_v257.py"], cwd=ROOT, check=True)
        require_memory(read(ROOT / MEMORY_PROOF))
        wait_gpu_slot()
        subprocess.run(plan["probe_command"], cwd=ROOT, check=True)
        report = read(probe / "training_report.json")
        require_result(report, probe=True)
        initial = check_initial(report, control["training_report"])
        write_new(ROOT / "benchmarks/v257_gpu_training_probe.json", {"purpose": "initial/runtime only, not score",
                  "report": report, "initial_validation_differences": initial})
        verify_bound_inputs(plan)
        wait_gpu_slot()
        subprocess.run(plan["command"], cwd=ROOT, check=True)
        report = read(output / "training_report.json")
        require_result(report)
        initial = check_initial(report, control["training_report"])
        verify_bound_inputs(plan)
        if data_record() != plan["dataset"]:
            raise RuntimeError("data changed during formal training")
        write_new(ROOT / "benchmarks/v257_initial_validation_check.json", {"matched": True,
                  "maximum_absolute_differences": initial})
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

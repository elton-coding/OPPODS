"""V254 full72k sixteen-expert architecture arm; storage arms are separate post-training work."""
from __future__ import annotations

import json
import os
import subprocess
import sys

from audit_pure_neural_candidate import fingerprint
from probe_pure_neural_sixteen_v254 import DESIGN, EVIDENCE, MAPPING, source_paths
from run_pure_neural_fairness_v251 import check_initial
from run_pure_neural_long_budget_v250 import command as control_command
from run_pure_neural_long_budget_v250 import data_record, require_report
from run_pure_neural_lr_v237 import ROOT, fingerprints
from run_pure_neural_pair_v253 import check_training_slot
from run_pure_neural_rms_budget_v242 import require_audit

CONTROL = "v250_eight_rms_72k"
LABEL = "v254_sixteen_rms_72k"
OUTPUT = "artifacts/pure_neural_v254/sixteen/steps72000"
PROBE = "artifacts/resource_probe/v254/sixteen_batch100"
MEMORY_PROOF = "benchmarks/v254_full_state_gpu_probe.json"


def command(*, probe=False):
    result = control_command()
    for flag, value in (("--model-design", DESIGN), ("--output-dir", PROBE if probe else OUTPUT),
                        ("--gpu-memory-fraction", "0.5")):
        result[result.index(flag) + 1] = value
    start = result.index("--baseline-expert-map") + 1
    result[start:start + 8] = list(map(str, MAPPING))
    if probe:
        for flag, value in (("--steps", "2"), ("--validate-every", "2")):
            result[result.index(flag) + 1] = value
    return result


def require_result(report, *, probe=False):
    steps = 2 if probe else 72000
    expected = {"requested_steps": steps, "batch_size": 100, "seed": 15240, "learning_rate": 1e-5,
                "loss_kind": "rms_score", "score_temperature": .5, "score_bce_weight": .05,
                "score_fairness_weight": .3, "quantile_bandwidth": .025,
                "train_components": ["transmitter", "receiver"], "validation_samples": 2000,
                "baseline_expert_map": MAPPING, "gpu_memory_fraction": .5,
                "trainable_parameters": 379103168, "variable_payload": False}
    history = [0, 2] if probe else list(range(0, 72001, 1000))
    if (any(report.get(key) != value for key, value in expected.items())
            or [row["step"] for row in report.get("history", [])] != history
            or report.get("gpu_peak_allocated_bytes", 0) <= 0):
        raise ValueError("V254 incomplete or mismatched sixteen-expert training")


def main():
    output, probe = ROOT / OUTPUT, ROOT / PROBE
    if output.exists() or probe.exists():
        raise FileExistsError("V254 output exists; refusing implicit optimizer restart")
    cpu = json.loads((ROOT / EVIDENCE).read_text(encoding="utf-8"))
    if cpu.get("matched") is not True or cpu.get("input_sha256") != fingerprints(source_paths()):
        raise ValueError("CPU initialization proof missing or stale")
    memory = json.loads((ROOT / MEMORY_PROOF).read_text(encoding="utf-8"))
    if (memory.get("passed") is not True or memory.get("batch_size") != 100
            or memory.get("route_counts") != [7] * 4 + [6] * 12
            or [row["adam_parameter_states"] for row in memory.get("steps", [])] != [2592, 2592]
            or memory.get("input_sha256") != fingerprints([ROOT / p for p in memory["input_sha256"]])):
        raise ValueError("all-expert Adam-state memory proof missing or stale")
    control_path = ROOT / f"benchmarks/{CONTROL}_audit_offset2000.json"
    control = json.loads(control_path.read_text(encoding="utf-8"))
    require_audit(control, CONTROL, 72000)
    require_report(control["training_report"], 72000)
    if fingerprint(ROOT / "artifacts/pure_neural_v250/eight/steps72000") != control["files"]:
        raise RuntimeError("frozen V250 control changed")
    if control["exact_mean_final"] >= 69:
        print("Local69 reached; defer new architecture training for frozen confirmation.", flush=True)
        return
    original_path = ROOT / "artifacts/pure_neural_v250/eight/execution_plan.json"
    original = json.loads(original_path.read_text(encoding="utf-8"))
    if fingerprints([ROOT / p for p in original["input_sha256"]]) != original["input_sha256"]:
        raise RuntimeError("original V250 training inputs changed")
    paths = source_paths() + [ROOT / p for p in original["input_sha256"]]
    paths += [ROOT / p for p in memory["input_sha256"]]
    paths += [control_path, original_path, ROOT / EVIDENCE, ROOT / MEMORY_PROOF]
    paths += [ROOT / p for p in (
        "scripts/run_pure_neural_sixteen_v254.py", "scripts/run_pure_neural_fairness_v251.py",
        "scripts/run_pure_neural_pair_v253.py", "scripts/run_pure_neural_routing_v252.py",
        "scripts/probe_pure_neural_pair_v253.py", "scripts/audit_pure_neural_candidate.py",
        "scripts/evaluate_submission.py", "scripts/compare_paired_holdout.py")]
    before, dataset = fingerprints(paths), data_record()
    active = check_training_slot()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / "execution.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        record = {"label": LABEL, "control": CONTROL, "command": command(), "probe_command": command(probe=True),
                  "input_sha256": before, "dataset": dataset, "train_channel_draws": 7200000,
                  "other_registered_training_pids": active, "initialization": "original V227; fresh Adam",
                  "single_factor": "eight5dB to sixteen2.5dB joint Tx/Rx experts; same full72k budget",
                  "resource_only_difference": "allocator cap0.4 to0.5; batch100/float32 unchanged",
                  "storage_factorial_pending_separate_work": True, "fp32_arm_not_automatically_deployable": True}
        with (output.parent / "execution_plan.json").open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
        print(json.dumps(record), flush=True)
        subprocess.run(record["probe_command"], cwd=ROOT, check=True)
        probe_report = json.loads((probe / "training_report.json").read_text(encoding="utf-8"))
        require_result(probe_report, probe=True)
        initial = check_initial(probe_report, control["training_report"])
        with (ROOT / "benchmarks/v254_gpu_training_probe.json").open("x", encoding="utf-8") as stream:
            json.dump({"purpose": "initial validation and runtime, not performance", "report": probe_report,
                       "initial_validation_differences": initial}, stream, indent=2)
        if fingerprints(paths) != before:
            raise RuntimeError("V254 inputs changed during probe")
        check_training_slot()
        subprocess.run(record["command"], cwd=ROOT, check=True)
        report = json.loads((output / "training_report.json").read_text(encoding="utf-8"))
        require_result(report)
        initial = check_initial(report, control["training_report"])
        if fingerprints(paths) != before or data_record() != dataset:
            raise RuntimeError("V254 frozen inputs changed during full training")
        with (ROOT / "benchmarks/v254_initial_validation_check.json").open("x", encoding="utf-8") as stream:
            json.dump({"maximum_absolute_differences": initial, "matched": True}, stream, indent=2)
        subprocess.run([sys.executable, "-u", "scripts/audit_pure_neural_candidate.py", "--submission", OUTPUT,
                        "--label", LABEL, "--baseline-label", CONTROL], cwd=ROOT, check=True)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()

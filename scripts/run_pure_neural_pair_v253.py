"""Wait for V250's released slot, then probe/train the matched-mass V253 pair partition."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import psutil
from audit_pure_neural_candidate import fingerprint
from probe_pure_neural_pair_v253 import DESIGN, EVIDENCE, MAPPING, source_paths
from run_pure_neural_fairness_v251 import check_initial
from run_pure_neural_long_budget_v250 import CONTROL, data_record, require_report
from run_pure_neural_lr_v237 import ROOT, fingerprints
from run_pure_neural_rms_budget_v242 import command as control_command
from run_pure_neural_rms_budget_v242 import require_audit
from run_pure_neural_routing_v252 import require_result

LABEL = "v253_pair_routing_rms_36k"
DEPENDENCY = "v250_eight_rms_72k"
REGION_CONTROL = "v252_balanced_routing_rms_36k"
OUTPUT = "artifacts/pure_neural_v253/eight/pair_steps36000"
PROBE = "artifacts/resource_probe/v253/pair_batch100"


def command(*, probe=False):
    result = control_command("eight")
    for flag, value in (("--model-design", DESIGN), ("--output-dir", PROBE if probe else OUTPUT)):
        result[result.index(flag) + 1] = value
    start = result.index("--baseline-expert-map") + 1
    result[start:start+8] = list(map(str, MAPPING))
    if probe:
        for flag, value in (("--steps", "2"), ("--validate-every", "2")):
            result[result.index(flag) + 1] = value
    return result


def wait_audit(label, steps, timeout):
    path = ROOT / f"benchmarks/{label}_audit_offset2000.json"
    deadline = time.monotonic() + timeout
    print(json.dumps({"waiting_for_audit": label, "expected_steps": steps}), flush=True)
    while True:
        if path.exists():
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
            else:
                require_audit(report, label, steps)
                if label == DEPENDENCY:
                    require_report(report["training_report"], 72000)
                    prefix = json.loads((ROOT / "benchmarks/v250_eight_rms_72k_prefix_check.json").read_text(encoding="utf-8"))
                    if prefix.get("matched") is not True or prefix.get("checkpoints") != 37:
                        raise ValueError("V250 must pass its registered full-prefix check")
                elif label == REGION_CONTROL:
                    require_result(report["training_report"])
                else:
                    raise ValueError("unregistered waiting dependency")
                return report
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{label} audit incomplete; inspect existing job, do not duplicate")
        time.sleep(min(30., max(.01, deadline - time.monotonic())))


def check_training_slot():
    """Known calibrate jobs are the GPU-heavy process type in this registered queue."""
    jobs = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = process.info["cmdline"] or []
            if (str(process.info["name"]).lower() == "python.exe" and "--gpu-memory-fraction" in cmd
                    and "--stage" in cmd and cmd[cmd.index("--stage")+1] == "calibrate"):
                jobs.append(process.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied, IndexError):
            continue
    if len(jobs) >= 2:
        raise RuntimeError(f"two registered GPU training jobs still active: {jobs}")
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-timeout", type=float, default=43200)
    args = parser.parse_args()
    output, probe = ROOT / OUTPUT, ROOT / PROBE
    if output.exists() or probe.exists():
        raise FileExistsError("V253 output exists; refusing implicit resume")
    cpu_path = ROOT / EVIDENCE
    cpu = json.loads(cpu_path.read_text(encoding="utf-8"))
    if cpu.get("matched") is not True or cpu.get("input_sha256") != fingerprints(source_paths()):
        raise ValueError("V253 CPU proof missing or stale")
    control_path = ROOT / f"benchmarks/{CONTROL}_audit_offset2000.json"
    control = json.loads(control_path.read_text(encoding="utf-8"))
    require_audit(control, CONTROL, 36000)
    require_report(control["training_report"], 36000)
    if fingerprint(ROOT / "artifacts/pure_neural_v242/eight/steps36000") != control["files"]:
        raise RuntimeError("V242E control changed")
    original_path = ROOT / "artifacts/pure_neural_v242/eight/execution_plan.json"
    original = json.loads(original_path.read_text(encoding="utf-8"))
    if fingerprints([ROOT / p for p in original["input_sha256"]]) != original["input_sha256"]:
        raise RuntimeError("original control inputs changed")
    paths = source_paths() + [ROOT / p for p in original["input_sha256"]] + [cpu_path, control_path, original_path]
    paths += [ROOT / p for p in ("ziliao/data_train/H_train.npz", "scripts/run_pure_neural_pair_v253.py",
              "scripts/run_pure_neural_long_budget_v250.py", "scripts/run_pure_neural_fairness_v251.py",
              "scripts/run_pure_neural_routing_v252.py", "scripts/run_pure_neural_rms_budget_v242.py",
              "scripts/audit_pure_neural_candidate.py", "scripts/evaluate_submission.py",
              "scripts/compare_paired_holdout.py", "artifacts/pure_neural_v252/eight/execution_plan.json")]
    before, dataset = fingerprints(paths), data_record()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / "execution.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        record = {"label": LABEL, "control": CONTROL, "region_control": REGION_CONTROL,
                  "dependency": DEPENDENCY, "command": command(), "probe_command": command(probe=True),
                  "input_sha256": before, "dataset": dataset, "train_channel_draws": 3600000,
                  "single_factor": "eight equal-mass min-only regions to four min bands split by conditional max median",
                  "initialization": "original V227; fresh Adam, not predecessor or probe checkpoint"}
        with (output.parent / "execution_plan.json").open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
        print(json.dumps(record), flush=True)
        dependency = wait_audit(DEPENDENCY, 72000, args.wait_timeout)
        observed_scores = [control["exact_mean_final"], dependency["exact_mean_final"]]
        region_path = ROOT / f"benchmarks/{REGION_CONTROL}_audit_offset2000.json"
        if region_path.exists():
            region = wait_audit(REGION_CONTROL, 36000, args.wait_timeout)
            observed_scores.append(region["exact_mean_final"])
        fairness_path = ROOT / "benchmarks/v251_eight_rms_fair50_36k_audit_offset2000.json"
        if fairness_path.exists():
            fairness = json.loads(fairness_path.read_text(encoding="utf-8"))
            require_audit(fairness, "v251_eight_rms_fair50_36k", 36000)
            observed_scores.append(fairness["exact_mean_final"])
        if max(observed_scores) >= 69:
            print("Local 69 reached; defer V253 for frozen confirmation.", flush=True)
            return
        if fingerprints(paths) != before or data_record() != dataset:
            raise RuntimeError("V253 inputs changed while waiting")
        if output.exists() or probe.exists():
            raise FileExistsError("V253 output appeared while waiting")
        active = check_training_slot()
        print(json.dumps({"other_registered_training_pids_before_probe": active}), flush=True)
        subprocess.run(record["probe_command"], cwd=ROOT, check=True)
        probe_report = json.loads((probe / "training_report.json").read_text(encoding="utf-8"))
        require_result(probe_report, probe=True)
        initial = check_initial(probe_report, control["training_report"])
        with (ROOT / "benchmarks/v253_gpu_training_probe.json").open("x", encoding="utf-8") as stream:
            json.dump({"purpose": "GPU resource/runtime only, not score evidence", "report": probe_report,
                       "initial_validation_differences": initial}, stream, indent=2)
        if fingerprints(paths) != before:
            raise RuntimeError("V253 inputs changed during probe")
        check_training_slot()
        subprocess.run(record["command"], cwd=ROOT, check=True)
        report = json.loads((output / "training_report.json").read_text(encoding="utf-8"))
        require_result(report)
        initial = check_initial(report, control["training_report"])
        if fingerprints(paths) != before or data_record() != dataset:
            raise RuntimeError("V253 inputs changed during formal training")
        with (ROOT / "benchmarks/v253_initial_validation_check.json").open("x", encoding="utf-8") as stream:
            json.dump({"maximum_absolute_differences": initial, "matched": True}, stream, indent=2)
        # V252 may still be training in the other slot; never inspect a partial checkpoint as a control.
        candidate_files = fingerprint(output)
        region = wait_audit(REGION_CONTROL, 36000, args.wait_timeout)
        if fingerprint(ROOT / "artifacts/pure_neural_v252/eight/balanced_steps36000") != region["files"]:
            raise RuntimeError("completed V252 control weights no longer match audit")
        if fingerprints(paths) != before or fingerprint(output) != candidate_files:
            raise RuntimeError("inputs or completed V253 weights changed while waiting for control")
        subprocess.run([sys.executable, "-u", "scripts/audit_pure_neural_candidate.py", "--submission", OUTPUT,
                        "--label", LABEL, "--baseline-label", CONTROL, "--control-label", REGION_CONTROL],
                       cwd=ROOT, check=True)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()

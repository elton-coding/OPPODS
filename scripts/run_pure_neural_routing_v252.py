"""Wait for V251, probe full batch100, then train isolated balanced routing 36k."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from audit_pure_neural_candidate import fingerprint
from probe_pure_neural_routing_v252 import DESIGN, EVIDENCE, MAPPING, source_paths
from run_pure_neural_fairness_v251 import LABEL as DEPENDENCY
from run_pure_neural_fairness_v251 import check_initial, require_candidate
from run_pure_neural_long_budget_v250 import CONTROL, data_record, require_report
from run_pure_neural_lr_v237 import ROOT, fingerprints
from run_pure_neural_rms_budget_v242 import command as control_command
from run_pure_neural_rms_budget_v242 import require_audit

LABEL = "v252_balanced_routing_rms_36k"
OUTPUT = "artifacts/pure_neural_v252/eight/balanced_steps36000"
PROBE = "artifacts/resource_probe/v252/balanced_batch100"


def command(*, probe=False):
    result = control_command("eight")
    for flag, value in (("--model-design", DESIGN), ("--output-dir", PROBE if probe else OUTPUT)):
        result[result.index(flag) + 1] = value
    start = result.index("--baseline-expert-map") + 1
    result[start:start + 8] = list(map(str, MAPPING))
    if probe:
        for flag, value in (("--steps", "2"), ("--validate-every", "2")):
            result[result.index(flag) + 1] = value
    return result


def require_result(report, *, probe=False):
    if report.get("baseline_expert_map") != MAPPING:
        raise ValueError("balanced routing requires the registered source mapping")
    if probe:
        if (report.get("requested_steps") != 2 or report.get("validation_samples") != 2000
                or [row["step"] for row in report.get("history", [])] != [0, 2]):
            raise ValueError("GPU probe must complete two steps at full batch100")
        normalized = {**report, "requested_steps": 36000, "validation_samples": 2000,
                      "history": [{"step": i} for i in range(0, 36001, 1000)]}
    else:
        normalized = report
    require_report({**normalized, "baseline_expert_map": [0, 0, 1, 1, 1, 1, 1, 1]}, 36000)


def wait_dependency(timeout):
    path = ROOT / f"benchmarks/{DEPENDENCY}_audit_offset2000.json"
    deadline = time.monotonic() + timeout
    print(json.dumps({"waiting_for_audit": DEPENDENCY, "expected_steps": 36000}), flush=True)
    while True:
        if path.exists():
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
            else:
                require_audit(report, DEPENDENCY, 36000)
                require_candidate(report["training_report"])
                return report
        if time.monotonic() >= deadline:
            raise TimeoutError("V251 audit incomplete; inspect existing job, do not duplicate training")
        time.sleep(min(30., max(.01, deadline - time.monotonic())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-timeout", type=float, default=43200)
    args = parser.parse_args()
    output, probe = ROOT / OUTPUT, ROOT / PROBE
    if output.exists() or probe.exists():
        raise FileExistsError("formal/probe output exists; refusing implicit resume")
    cpu_path = ROOT / EVIDENCE
    cpu = json.loads(cpu_path.read_text(encoding="utf-8"))
    if not cpu.get("matched") or cpu.get("input_sha256") != fingerprints(source_paths()):
        raise ValueError("CPU initialization probe missing or stale")
    control_path = ROOT / f"benchmarks/{CONTROL}_audit_offset2000.json"
    control = json.loads(control_path.read_text(encoding="utf-8"))
    require_audit(control, CONTROL, 36000)
    require_report(control["training_report"], 36000)
    if fingerprint(ROOT / "artifacts/pure_neural_v242/eight/steps36000") != control["files"]:
        raise RuntimeError("frozen control changed")
    original_path = ROOT / "artifacts/pure_neural_v242/eight/execution_plan.json"
    original = json.loads(original_path.read_text(encoding="utf-8"))
    if fingerprints([ROOT / p for p in original["input_sha256"]]) != original["input_sha256"]:
        raise RuntimeError("original V242 control inputs changed")
    paths = source_paths() + [ROOT / p for p in original["input_sha256"]] + [control_path, cpu_path, original_path]
    paths += [ROOT / p for p in ("ziliao/data_train/H_train.npz", "scripts/run_pure_neural_routing_v252.py",
              "scripts/run_pure_neural_long_budget_v250.py", "scripts/run_pure_neural_fairness_v251.py",
              "scripts/run_pure_neural_rms_budget_v242.py", "scripts/audit_pure_neural_candidate.py",
              "scripts/evaluate_submission.py", "scripts/compare_paired_holdout.py")]
    before, dataset = fingerprints(paths), data_record()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / "execution.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        record = {"label": LABEL, "control": CONTROL, "command": command(), "probe_command": command(probe=True),
                  "input_sha256": before, "dataset": dataset, "dependency": DEPENDENCY,
                  "single_factor": "min-SNR partition boundaries; parent mapping preserves initial function",
                  "train_channel_draws": 3600000, "initialization": "original V227, fresh Adam"}
        with (output.parent / "execution_plan.json").open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
        print(json.dumps(record), flush=True)
        dependency = wait_dependency(args.wait_timeout)
        if max(control["exact_mean_final"], dependency["exact_mean_final"]) >= 69:
            print("Local 69 reached; defer extra training for frozen confirmation.", flush=True)
            return
        if fingerprints(paths) != before or data_record() != dataset:
            raise RuntimeError("registered inputs changed while waiting")
        if output.exists() or probe.exists():
            raise FileExistsError("output appeared while waiting")
        subprocess.run(record["probe_command"], cwd=ROOT, check=True)
        probe_report = json.loads((probe / "training_report.json").read_text(encoding="utf-8"))
        require_result(probe_report, probe=True)
        probe_initial = check_initial(probe_report, control["training_report"])
        with (ROOT / "benchmarks/v252_gpu_training_probe.json").open("x", encoding="utf-8") as stream:
            json.dump({"purpose": "full-batch resource/runtime only, not score evidence", "report": probe_report,
                       "initial_validation_differences": probe_initial}, stream, indent=2)
        if fingerprints(paths) != before:
            raise RuntimeError("registered inputs changed during probe")
        subprocess.run(record["command"], cwd=ROOT, check=True)
        report = json.loads((output / "training_report.json").read_text(encoding="utf-8"))
        require_result(report)
        differences = check_initial(report, control["training_report"])
        if fingerprints(paths) != before or data_record() != dataset:
            raise RuntimeError("registered inputs changed during formal training")
        with (ROOT / "benchmarks/v252_initial_validation_check.json").open("x", encoding="utf-8") as stream:
            json.dump({"maximum_absolute_differences": differences, "matched": True}, stream, indent=2)
        subprocess.run([sys.executable, "-u", "scripts/audit_pure_neural_candidate.py", "--submission", OUTPUT,
                        "--label", LABEL, "--baseline-label", CONTROL], cwd=ROOT, check=True)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()

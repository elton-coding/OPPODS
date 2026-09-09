"""Batch x learning-rate factorial completion, fresh V227 and matched channel budget."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from run_pure_neural_global_batch_v246 import command as batch_command
from run_pure_neural_global_batch_v246 import require_report
from run_pure_neural_lr_v237 import ROOT, fingerprints

LABEL = "v247_batch400_lr5"
DEPENDENCY = "v246_global_batch400"
OUTPUT = "artifacts/pure_neural_v247/batch400_lr5"
PROBE = "artifacts/resource_probe/v247/cuda_batch400_lr5"


def command(*, probe=False):
    result = batch_command(probe=probe)
    for flag, value in (("--learning-rate", "5e-5"), ("--output-dir", PROBE if probe else OUTPUT)):
        result[result.index(flag) + 1] = value
    return result


def require_dependency(report):
    require_report(report["training_report"])
    if (report.get("label") != DEPENDENCY or len(report.get("per_seed", [])) != 3
            or report.get("protocol") != {"split_seed": 1176, "test_offset": 2000,
                                         "samples": 2000, "noise_seeds": [22701, 22702, 22703]}
            or report["training_report"].get("learning_rate") != 1e-5):
        raise ValueError("V246 full 3000-step fixed audit required")


def wait_dependency(timeout):
    deadline = time.monotonic() + timeout
    path = ROOT / f"benchmarks/{DEPENDENCY}_audit_offset2000.json"
    print(json.dumps({"waiting_for_audit": DEPENDENCY, "expected_steps": 3000}), flush=True)
    while True:
        if path.exists():
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
            else:
                require_dependency(report)
                return report
        if time.monotonic() >= deadline:
            raise TimeoutError("V246 incomplete: inspect existing process; do not restart")
        time.sleep(min(30., max(.01, deadline - time.monotonic())))


def factorial_command():
    result = [sys.executable, "scripts/compare_factorial_holdout.py"]
    for arm, label in (("c", "v230_control"), ("a", "v237_two_lr5"),
                       ("b", DEPENDENCY), ("ab", LABEL)):
        result += [f"--{arm}", *[f"benchmarks/{label}_holdout1176_offset2000_noise{seed}.npz"
                                 for seed in (22701, 22702, 22703)]]
    return [*result, "--factor-a", "learning_rate_5e-5",
            "--factor-b", "batch400_at_1.2M_channel_draws",
            "--output", "benchmarks/v247_batch_lr_factorial.json"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-timeout", type=float, default=21600)
    args = parser.parse_args()
    output, probe = ROOT / OUTPUT, ROOT / PROBE
    if output.exists() or probe.exists():
        raise FileExistsError("existing formal/probe output: refusing implicit resume")
    cpu = json.loads((ROOT / "benchmarks/v246_parent_cpu_gradient_probe.json").read_text(encoding="utf-8"))
    paths = [ROOT / name for name in cpu["input_sha256"]]
    if (fingerprints(paths) != cpu["input_sha256"] or cpu["max_gradient_delta"] > 1e-7
            or cpu["max_logit_delta"] != 0 or not cpu["backward_preserves_generator"]):
        raise RuntimeError("reuse only the verified, unchanged checkpointed link")
    paths += [ROOT / name for name in ("scripts/run_pure_neural_global_batch_v246.py",
              "scripts/run_pure_neural_lr_v237.py", "scripts/run_pure_neural_batch_lr_v247.py",
              "benchmarks/v230_control_audit_offset2000.json", "benchmarks/v237_two_lr5_audit_offset2000.json")]
    before = fingerprints(paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / "execution.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        record = {"label": LABEL, "dependency": DEPENDENCY, "output": OUTPUT,
                  "command": command(), "probe_command": command(probe=True), "input_sha256": before,
                  "factorial_command": factorial_command(), "channel_draw_budget": 1200000}
        with (output.parent / "execution_plan.json").open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
        dependency = wait_dependency(args.wait_timeout)
        if dependency["exact_mean_final"] >= 69:
            print("Local 69 reached; defer LR combination for frozen confirmation.", flush=True)
            return
        if fingerprints(paths) != before or output.exists() or probe.exists():
            raise RuntimeError("inputs changed or output appeared while waiting")
        print(json.dumps(record), flush=True)
        subprocess.run(record["probe_command"], cwd=ROOT, check=True)
        probe_report = json.loads((probe / "training_report.json").read_text(encoding="utf-8"))
        require_report(probe_report, probe=True)
        if probe_report["learning_rate"] != 5e-5:
            raise ValueError("wrong probe LR")
        with (ROOT / "benchmarks/v247_gpu_training_probe.json").open("x", encoding="utf-8") as stream:
            json.dump({"purpose": "resource/runtime only, not score evidence", "report": probe_report}, stream, indent=2)
        if fingerprints(paths) != before:
            raise RuntimeError("inputs changed in probe")
        subprocess.run(record["command"], cwd=ROOT, check=True)
        report = json.loads((output / "training_report.json").read_text(encoding="utf-8"))
        require_report(report)
        if report["learning_rate"] != 5e-5 or fingerprints(paths) != before:
            raise RuntimeError("LR or inputs changed; do not audit")
        for metric in ("loss", "efficiency", "fairness", "final"):
            if abs(report["history"][0][metric] - dependency["training_report"]["history"][0][metric]) > 1e-4:
                raise ValueError("initial validation does not match V246")
        subprocess.run([sys.executable, "-u", "scripts/audit_pure_neural_candidate.py",
                        "--submission", OUTPUT, "--label", LABEL,
                        "--baseline-label", "v240_two_36k", "--control-label", DEPENDENCY], cwd=ROOT, check=True)
        if (ROOT / "benchmarks/v247_batch_lr_factorial.json").exists():
            raise FileExistsError("factorial evidence already exists")
        subprocess.run(record["factorial_command"], cwd=ROOT, check=True)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()

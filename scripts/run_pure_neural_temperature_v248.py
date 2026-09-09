"""Single-factor T=1 trial, after completing V242's eight-expert factorial audit."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from run_pure_neural_lr_v237 import ROOT, fingerprints, train_command
from run_pure_neural_rms_budget_v242 import require_audit

DEPENDENCY = "v242_eight_rms_36k"
PLAN = {"label": "v248_temperature1", "design": "research/pure_neural_v227/modelDesign.py",
        "parent": "artifacts/pure_neural_v227/joint_low", "mapping": [0, 1],
        "output": "artifacts/pure_neural_v248/temperature1", "learning_rate": "1e-5"}
PROBE = "artifacts/resource_probe/v248/temperature1"


def command(*, probe=False):
    result = train_command(PLAN)
    result[result.index("--score-temperature") + 1] = "1.0"
    if probe:
        for flag, value in (("--output-dir", PROBE), ("--steps", "2"),
                            ("--validate-every", "2"), ("--validation-samples", "4")):
            result[result.index(flag) + 1] = value
    return result


def require_report(report, *, probe=False):
    steps = 2 if probe else 12000
    expected = {"requested_steps": steps, "score_temperature": 1., "score_bce_weight": .05,
                "score_fairness_weight": .3, "quantile_bandwidth": .025,
                "loss_kind": "soft_score", "batch_size": 100, "learning_rate": 1e-5,
                "seed": 15240, "baseline_expert_map": [0, 1],
                "train_components": ["transmitter", "receiver"],
                "validation_samples": 4 if probe else 2000}
    if (any(report.get(key) != value for key, value in expected.items())
            or not report.get("history") or report["history"][-1]["step"] != steps
            or report.get("gpu_peak_allocated_bytes", 0) <= 0):
        raise ValueError("incomplete or mismatched T1 experiment")


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
                return report
        if time.monotonic() >= deadline:
            raise TimeoutError("V242 incomplete: inspect existing job, do not restart")
        time.sleep(min(30., max(.01, deadline - time.monotonic())))


def score_paths(label):
    return [f"benchmarks/{label}_holdout1176_offset2000_noise{seed}.npz"
            for seed in (22701, 22702, 22703)]


def predecessor_commands():
    paired = [sys.executable, "scripts/compare_paired_holdout.py",
              "--baseline", *score_paths("v240_eight_36k"),
              "--candidate", *score_paths(DEPENDENCY),
              "--output", "benchmarks/v242_eight_rms_vs_soft_36k_paired.json"]
    factorial = [sys.executable, "scripts/compare_factorial_holdout.py"]
    for arm, label in (("c", "v230_eight"), ("a", "v241_eight_rms"),
                       ("b", "v240_eight_36k"), ("ab", DEPENDENCY)):
        factorial += [f"--{arm}", *score_paths(label)]
    factorial += ["--factor-a", "rms_loss", "--factor-b", "budget_36k",
                  "--output", "benchmarks/v242_eight_loss_budget_factorial.json"]
    return [paired, factorial]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-timeout", type=float, default=21600)
    args = parser.parse_args()
    output, probe = ROOT / PLAN["output"], ROOT / PROBE
    if output.exists() or probe.exists():
        raise FileExistsError("existing output: refusing implicit resume")
    paths = [ROOT / PLAN["design"], *[ROOT / PLAN["parent"] / name for name in
             ("encoder.pth", "transmitter.pth", "receiver.pth")]]
    paths += [ROOT / name for name in ("scripts/train_pure_neural_snr_experts.py",
              "scripts/run_pure_neural_lr_v237.py", "scripts/run_pure_neural_rms_budget_v242.py",
              "scripts/run_pure_neural_temperature_v248.py", "scripts/compare_factorial_holdout.py",
              "scripts/compare_paired_holdout.py", "benchmarks/v230_control_audit_offset2000.json")]
    before = fingerprints(paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / "execution.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        record = {**PLAN, "input_sha256": before, "command": command(), "probe_command": command(probe=True),
                  "dependency": DEPENDENCY, "predecessor_commands": predecessor_commands()}
        with (output.parent / "execution_plan.json").open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
        dependency = wait_dependency(args.wait_timeout)
        if fingerprints(paths) != before:
            raise RuntimeError("registered inputs changed while waiting")
        # Finish the predecessor's same-budget and factorial analyses without GPU use.
        for cmd in record["predecessor_commands"]:
            if (ROOT / cmd[-1]).exists():
                raise FileExistsError("predecessor evidence exists: inspect rather than overwrite")
            subprocess.run(cmd, cwd=ROOT, check=True)
        if dependency["exact_mean_final"] >= 69:
            print("Local 69 reached; defer T1 trial for frozen confirmation.", flush=True)
            return
        if output.exists() or probe.exists():
            raise FileExistsError("output appeared while waiting")
        print(json.dumps(record), flush=True)
        subprocess.run(record["probe_command"], cwd=ROOT, check=True)
        probe_report = json.loads((probe / "training_report.json").read_text(encoding="utf-8"))
        require_report(probe_report, probe=True)
        with (ROOT / "benchmarks/v248_gpu_training_probe.json").open("x", encoding="utf-8") as stream:
            json.dump({"purpose": "runtime/resource only, not score evidence", "report": probe_report}, stream, indent=2)
        if fingerprints(paths) != before:
            raise RuntimeError("inputs changed during probe")
        subprocess.run(record["command"], cwd=ROOT, check=True)
        report = json.loads((output / "training_report.json").read_text(encoding="utf-8"))
        require_report(report)
        control = json.loads((ROOT / "benchmarks/v230_control_audit_offset2000.json").read_text(encoding="utf-8"))
        for metric in ("loss", "efficiency", "fairness", "final"):
            if abs(report["history"][0][metric] - control["training_report"]["history"][0][metric]) > 1e-4:
                raise ValueError("T1 initial validation differs from control")
        if fingerprints(paths) != before:
            raise RuntimeError("registered training inputs changed; do not audit")
        subprocess.run([sys.executable, "-u", "scripts/audit_pure_neural_candidate.py",
                        "--submission", PLAN["output"], "--label", PLAN["label"],
                        "--baseline-label", "v240_two_36k", "--control-label", "v230_control"], cwd=ROOT, check=True)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()

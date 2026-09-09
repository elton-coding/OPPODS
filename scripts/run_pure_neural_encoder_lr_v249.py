"""Encoder-only learning-rate ablation versus V235; queue behind complete V247."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from run_pure_neural_global_batch_v246 import require_report as require_batch_report
from run_pure_neural_lr_v237 import ROOT, fingerprints, train_command
from run_pure_neural_rms_budget_v242 import PROTOCOL

DEPENDENCY = "v247_batch400_lr5"
PLAN = {"label": "v249_encoder_lr5", "design": "research/pure_neural_v227/modelDesign.py",
        "parent": "artifacts/pure_neural_v227/joint_low", "mapping": [0, 1],
        "output": "artifacts/pure_neural_v249/encoder_lr5", "learning_rate": "1e-5"}
PROBE = "artifacts/resource_probe/v249/encoder_lr5"


def command(*, probe=False):
    result = train_command(PLAN)
    result[2] = "scripts/train_pure_neural_encoder_lr_v249.py"
    result.insert(result.index("--train-components") + 1, "encoder")
    if probe:
        for flag, value in (("--output-dir", PROBE), ("--steps", "2"),
                            ("--validate-every", "2"), ("--validation-samples", "4")):
            result[result.index(flag) + 1] = value
    return result


def require_dependency(report):
    require_batch_report(report["training_report"])
    if (report.get("label") != DEPENDENCY or report.get("protocol") != PROTOCOL
            or len(report.get("per_seed", [])) != 3 or report["training_report"].get("learning_rate") != 5e-5):
        raise ValueError("expected V247 completed 3000-step fixed audit")


def wait_dependency(timeout):
    deadline = time.monotonic() + timeout
    path = ROOT / f"benchmarks/{DEPENDENCY}_audit_offset2000.json"
    print(json.dumps({"waiting_for_audit": DEPENDENCY, "expected_steps": 3000}), flush=True)
    while True:
        if path.exists():
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
            else:
                require_dependency(result)
                return result
        if time.monotonic() >= deadline:
            raise TimeoutError("V247 incomplete: inspect existing task, do not restart")
        time.sleep(min(30., max(.01, deadline - time.monotonic())))


def require_report(report, *, probe=False):
    steps = 2 if probe else 12000
    expected = {"requested_steps": steps, "train_components": ["encoder", "transmitter", "receiver"],
                "trainable_parameters": 48191288, "batch_size": 100, "learning_rate": 1e-5,
                "encoder_learning_rate": 5e-5, "transceiver_learning_rate": 1e-5,
                "baseline_expert_map": [0, 1], "seed": 15240, "loss_kind": "soft_score",
                "score_temperature": .5, "score_bce_weight": .05, "score_fairness_weight": .3,
                "quantile_bandwidth": .025, "validation_samples": 4 if probe else 2000}
    groups = [{"name": "encoder", "lr": 5e-5, "parameters": 803392},
              {"name": "transceiver", "lr": 1e-5, "parameters": 47387896}]
    if (any(report.get(key) != value for key, value in expected.items())
            or report.get("optimizer_parameter_groups") != groups or not report.get("history")
            or report["history"][-1]["step"] != steps or report.get("gpu_peak_allocated_bytes", 0) <= 0):
        raise ValueError("incomplete or mismatched two-rate Adam report")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-timeout", type=float, default=21600)
    args = parser.parse_args()
    output, probe = ROOT / PLAN["output"], ROOT / PROBE
    if output.exists() or probe.exists():
        raise FileExistsError("existing output: refusing implicit resume")
    cpu_path = ROOT / "benchmarks/v249_cpu_runtime_probe.json"
    cpu = json.loads(cpu_path.read_text(encoding="utf-8"))
    cpu_paths = [ROOT / name for name in cpu["input_sha256"]]
    cpu_hashes = {str((ROOT / name).relative_to(ROOT)): value for name, value in cpu["input_sha256"].items()}
    if (fingerprints(cpu_paths) != cpu_hashes or cpu["report"]["history"][-1]["step"] != 2
            or cpu["report"].get("encoder_learning_rate") != 5e-5):
        raise RuntimeError("require the unchanged CPU-tested two-rate training entry")
    paths = [ROOT / PLAN["design"], *[ROOT / PLAN["parent"] / name for name in
             ("encoder.pth", "transmitter.pth", "receiver.pth")]]
    paths += [ROOT / name for name in ("scripts/train_pure_neural_snr_experts.py",
              "scripts/train_pure_neural_encoder_lr_v249.py", "scripts/run_pure_neural_encoder_lr_v249.py",
              "scripts/run_pure_neural_lr_v237.py", "scripts/run_pure_neural_global_batch_v246.py",
              "scripts/run_pure_neural_rms_budget_v242.py", "benchmarks/v235_end_to_end_audit_offset2000.json")]
    paths.append(cpu_path)
    before = fingerprints(paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / "execution.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        record = {**PLAN, "input_sha256": before, "command": command(), "probe_command": command(probe=True),
                  "dependency": DEPENDENCY, "control": "v235_end_to_end", "encoder_lr_multiplier": 5.}
        with (output.parent / "execution_plan.json").open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
        dependency = wait_dependency(args.wait_timeout)
        if dependency["exact_mean_final"] >= 69:
            print("Local 69 reached; defer encoder-LR trial for frozen confirmation.", flush=True)
            return
        if fingerprints(paths) != before or output.exists() or probe.exists():
            raise RuntimeError("inputs changed or output appeared while waiting")
        print(json.dumps(record), flush=True)
        subprocess.run(record["probe_command"], cwd=ROOT, check=True)
        probe_report = json.loads((probe / "training_report.json").read_text(encoding="utf-8"))
        require_report(probe_report, probe=True)
        with (ROOT / "benchmarks/v249_gpu_training_probe.json").open("x", encoding="utf-8") as stream:
            json.dump({"purpose": "resource/runtime only, not score evidence", "report": probe_report}, stream, indent=2)
        if fingerprints(paths) != before:
            raise RuntimeError("inputs changed during probe")
        subprocess.run(record["command"], cwd=ROOT, check=True)
        report = json.loads((output / "training_report.json").read_text(encoding="utf-8"))
        require_report(report)
        control = json.loads((ROOT / "benchmarks/v235_end_to_end_audit_offset2000.json").read_text(encoding="utf-8"))
        for metric in ("loss", "efficiency", "fairness", "final"):
            if abs(report["history"][0][metric] - control["training_report"]["history"][0][metric]) > 1e-4:
                raise ValueError("initial validation differs from V235")
        if fingerprints(paths) != before:
            raise RuntimeError("training inputs changed; do not audit")
        subprocess.run([sys.executable, "-u", "scripts/audit_pure_neural_candidate.py",
                        "--submission", PLAN["output"], "--label", PLAN["label"],
                        "--baseline-label", "v240_two_36k", "--control-label", "v235_end_to_end"], cwd=ROOT, check=True)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()

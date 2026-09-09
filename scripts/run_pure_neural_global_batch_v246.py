"""Resource-gated batch400/micro100 ablation at the control's 1.2M channel draws."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from run_pure_neural_lr_v237 import ROOT, fingerprints, train_command, wait_for_audit

PLAN = {"label": "v246_global_batch400", "design": "research/pure_neural_v227/modelDesign.py",
        "parent": "artifacts/pure_neural_v227/joint_low", "mapping": [0, 1],
        "output": "artifacts/pure_neural_v246/batch400", "learning_rate": "1e-5"}
DEPENDENCY = "v245_bce_low"


def command(*, probe=False):
    result = train_command(PLAN)
    result[2] = "scripts/train_pure_neural_global_batch_v246.py"
    for flag, value in (("--batch-size", "400"), ("--steps", "2" if probe else "3000"),
                        ("--validate-every", "2" if probe else "250"),
                        ("--validation-samples", "4" if probe else "2000"),
                        ("--output-dir", "artifacts/resource_probe/v246/cuda_batch400" if probe else PLAN["output"])):
        result[result.index(flag) + 1] = value
    return [*result, "--microbatch-size", "100"]


def require_report(report, *, probe=False):
    steps = 2 if probe else 3000
    if (report.get("requested_steps") != steps or not report.get("history")
            or report["history"][-1]["step"] != steps or report.get("batch_size") != 400
            or report.get("microbatch_size") != 100 or report.get("validation_batch_size") != 100
            or report.get("global_loss_ue_count") != 800 or report.get("training_channel_draws") != 400 * steps
            or report.get("loss_kind") != "soft_score" or report.get("score_bce_weight") != .05
            or report.get("gpu_peak_allocated_bytes", 0) <= 0):
        raise ValueError("incomplete or mismatched global-batch run")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-timeout", type=float, default=21600)
    args = parser.parse_args()
    output = ROOT / PLAN["output"]
    gpu_probe = ROOT / "artifacts/resource_probe/v246/cuda_batch400"
    if output.exists() or gpu_probe.exists():
        raise FileExistsError("existing formal/probe output; refusing implicit resume")
    cpu = json.loads((ROOT / "benchmarks/v246_parent_cpu_gradient_probe.json").read_text(encoding="utf-8"))
    if cpu["max_logit_delta"] != 0 or cpu["max_gradient_delta"] > 1e-7 or not cpu["backward_preserves_generator"]:
        raise ValueError("real parent CPU equivalence required")
    paths = [ROOT / name for name in cpu["input_sha256"]]
    if fingerprints(paths) != cpu["input_sha256"]:
        raise RuntimeError("CPU-tested inputs changed")
    paths += [ROOT / "scripts/run_pure_neural_global_batch_v246.py", ROOT / "scripts/run_pure_neural_lr_v237.py"]
    before = fingerprints(paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / "execution.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        record = {**PLAN, "command": command(), "probe_command": command(probe=True),
                  "input_sha256": before, "dependency": DEPENDENCY,
                  "control": "v230_control", "channel_draw_budget": 1200000}
        with (output.parent / "execution_plan.json").open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
        wait_for_audit(DEPENDENCY, args.wait_timeout)
        dependency = json.loads((ROOT / f"benchmarks/{DEPENDENCY}_audit_offset2000.json").read_text(encoding="utf-8"))
        if dependency["exact_mean_final"] >= 69:
            print("Local 69 reached; defer batch change for frozen confirmation.", flush=True)
            return
        if fingerprints(paths) != before or output.exists() or gpu_probe.exists():
            raise RuntimeError("inputs changed or output appeared during wait")
        print(json.dumps(record), flush=True)
        subprocess.run(record["probe_command"], cwd=ROOT, check=True)
        report = json.loads((gpu_probe / "training_report.json").read_text(encoding="utf-8"))
        require_report(report, probe=True)
        with (ROOT / "benchmarks/v246_gpu_training_probe.json").open("x", encoding="utf-8") as stream:
            json.dump({"purpose": "runtime/resource probe only, not performance evidence", "report": report}, stream, indent=2)
        if fingerprints(paths) != before:
            raise RuntimeError("inputs changed during probe")
        subprocess.run(record["command"], cwd=ROOT, check=True)
        report = json.loads((output / "training_report.json").read_text(encoding="utf-8"))
        require_report(report)
        control = json.loads((ROOT / "benchmarks/v230_control_audit_offset2000.json").read_text(encoding="utf-8"))
        for metric in ("loss", "efficiency", "fairness", "final"):
            if abs(report["history"][0][metric] - control["training_report"]["history"][0][metric]) > 1e-4:
                raise RuntimeError("initial fixed validation does not match control")
        if fingerprints(paths) != before:
            raise RuntimeError("inputs changed during training; do not audit")
        subprocess.run([sys.executable, "-u", "scripts/audit_pure_neural_candidate.py",
                        "--submission", PLAN["output"], "--label", PLAN["label"],
                        "--baseline-label", "v240_two_36k", "--control-label", "v230_control"],
                       cwd=ROOT, check=True)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()

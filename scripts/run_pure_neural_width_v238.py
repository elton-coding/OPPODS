"""Wait for a free training slot, probe width-1024 memory, train and audit V238."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from run_pure_neural_lr_v237 import ROOT, fingerprints, require_complete, train_command, wait_for_audit


def plan() -> dict:
    return {
        "arm": "wide", "label": "v238_width",
        "design": "research/pure_neural_v238/modelDesign.py",
        "parent": "artifacts/pure_neural_v227/joint_low", "mapping": [0, 1],
        "output": "artifacts/pure_neural_v238/wide", "learning_rate": "1e-5",
    }


def probe_command(experiment: dict) -> list[str]:
    command = train_command(experiment)
    for flag, value in (("--output-dir", "artifacts/resource_probe/v238/gpu_batch100"),
                        ("--steps", "2"), ("--validation-samples", "4"), ("--validate-every", "2")):
        command[command.index(flag) + 1] = value
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-audit", default="v237_two_lr5")
    args = parser.parse_args()
    experiment = plan()
    output = ROOT / experiment["output"]
    probe_output = ROOT / "artifacts/resource_probe/v238/gpu_batch100"
    if output.exists() or probe_output.exists():
        raise FileExistsError("probe or formal output already exists; refusing implicit resume")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / "execution.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        wait_for_audit(args.wait_for_audit, timeout=10800)
        dependency = json.loads((ROOT / "benchmarks" / f"{args.wait_for_audit}_audit_offset2000.json")
                                .read_text(encoding="utf-8"))
        if dependency["exact_mean_final"] >= 69:
            print("Dependency reached local 69; defer width training for frozen confirmation.", flush=True)
            return
        paths = [ROOT / experiment["design"], ROOT / "scripts/train_pure_neural_snr_experts.py"]
        paths += [ROOT / experiment["parent"] / name
                  for name in ("encoder.pth", "transmitter.pth", "receiver.pth")]
        before = fingerprints(paths)
        record = {**experiment, "dependency": args.wait_for_audit, "input_sha256": before,
                  "probe_command": probe_command(experiment), "command": train_command(experiment)}
        (output.parent / "execution_plan.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(json.dumps(record), flush=True)
        subprocess.run(record["probe_command"], cwd=ROOT, check=True)
        probe = json.loads((probe_output / "training_report.json").read_text(encoding="utf-8"))
        if (probe["requested_steps"] != 2 or probe["history"][-1]["step"] != 2
                or probe["batch_size"] != 100 or probe["gpu_peak_allocated_bytes"] <= 0):
            raise RuntimeError("GPU batch100 resource probe did not complete")
        if fingerprints(paths) != before or output.exists():
            raise RuntimeError("inputs changed or formal output unexpectedly exists")
        subprocess.run(record["command"], cwd=ROOT, check=True)
        require_complete(json.loads((output / "training_report.json").read_text(encoding="utf-8")))
        if fingerprints(paths) != before:
            raise RuntimeError("training inputs changed; do not audit")
        subprocess.run([
            sys.executable, "-u", "scripts/audit_pure_neural_candidate.py",
            "--submission", experiment["output"], "--label", experiment["label"],
            "--baseline-label", "v230_eight", "--control-label", "v230_control",
        ], cwd=ROOT, check=True)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()

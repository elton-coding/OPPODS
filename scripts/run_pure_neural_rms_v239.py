"""Run V239 after the eight-expert LR audit frees a GPU training slot."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from run_pure_neural_lr_v237 import ROOT, fingerprints, require_complete, train_command, wait_for_audit

PLAN = {
    "arm": "rms", "label": "v239_rms_score",
    "design": "research/pure_neural_v227/modelDesign.py",
    "parent": "artifacts/pure_neural_v227/joint_low", "mapping": [0, 1],
    "output": "artifacts/pure_neural_v239/rms_score", "learning_rate": "1e-5",
}


def command() -> list[str]:
    result = train_command(PLAN)
    result[2] = "scripts/train_pure_neural_rms_v239.py"
    result[result.index("--loss-kind") + 1] = "rms_score"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-audit", default="v237_eight_lr5")
    args = parser.parse_args()
    output = ROOT / PLAN["output"]
    if output.exists():
        raise FileExistsError("formal output already exists; refusing implicit resume")
    probe_path = ROOT / "artifacts/resource_probe/v239/cpu_adapter/training_report.json"
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    if probe.get("loss_kind") != "rms_score" or probe["history"][-1]["step"] != 2:
        raise ValueError("CPU training adapter probe must complete first")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / "execution.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        wait_for_audit(args.wait_for_audit, timeout=10800)
        dependency = json.loads((ROOT / "benchmarks" / f"{args.wait_for_audit}_audit_offset2000.json")
                                .read_text(encoding="utf-8"))
        if dependency["exact_mean_final"] >= 69:
            print("Dependency reached local 69; defer new loss for frozen confirmation.", flush=True)
            return
        if output.exists():
            raise FileExistsError("formal output appeared while waiting")
        paths = [ROOT / PLAN["design"], ROOT / "scripts/train_pure_neural_snr_experts.py",
                 ROOT / "scripts/train_pure_neural_rms_v239.py"]
        paths += [ROOT / PLAN["parent"] / name for name in ("encoder.pth", "transmitter.pth", "receiver.pth")]
        before = fingerprints(paths)
        record = {**PLAN, "command": command(), "input_sha256": before, "dependency": args.wait_for_audit,
                  "normalization": "differentiable per-UE RMS, minimum 1e-4; BCE on original logits"}
        (output.parent / "execution_plan.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(json.dumps(record), flush=True)
        subprocess.run(record["command"], cwd=ROOT, check=True)
        report = json.loads((output / "training_report.json").read_text(encoding="utf-8"))
        require_complete(report)
        if report["loss_kind"] != "rms_score" or fingerprints(paths) != before:
            raise RuntimeError("objective or inputs changed; do not audit")
        subprocess.run([
            sys.executable, "-u", "scripts/audit_pure_neural_candidate.py", "--submission", PLAN["output"],
            "--label", PLAN["label"], "--baseline-label", "v230_eight", "--control-label", "v230_control",
        ], cwd=ROOT, check=True)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()

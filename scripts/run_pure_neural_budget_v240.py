"""Nested 12k-to-36k training-budget ablation with unchanged initial RNG and Adam."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from run_pure_neural_lr_v237 import ROOT, experiment, fingerprints, train_command, wait_for_audit


def plan(arm: str) -> dict:
    result = experiment(arm)
    result.update(label=f"v240_{arm}_36k", learning_rate="1e-5",
                  output=f"artifacts/pure_neural_v240/{arm}/steps36000")
    return result


def command(arm: str) -> list[str]:
    result = train_command(plan(arm))
    result[result.index("--steps") + 1] = "36000"
    result[result.index("--patience") + 1] = "36"
    return result


def check_prefix(report: dict, control: dict) -> dict:
    prefix = [row for row in report["history"] if row["step"] <= 12000]
    if [row["step"] for row in prefix] != [row["step"] for row in control["history"]]:
        raise ValueError("12k prefix validation checkpoints do not match control")
    errors = {key: max(abs(float(a[key]) - float(b[key])) for a, b in zip(prefix, control["history"], strict=True))
              for key in ("loss", "efficiency", "fairness", "final")}
    matched = errors["loss"] <= 1e-5 and all(errors[key] <= 1e-4 for key in ("efficiency", "fairness", "final"))
    return {"matched": matched, "maximum_absolute_difference": errors,
            "scope": "same 0..12000-step validation trajectory, not a proof of bitwise optimizer-state equality"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("two", "eight"), required=True)
    parser.add_argument("--wait-for-audit")
    args = parser.parse_args()
    config = plan(args.arm)
    output = ROOT / config["output"]
    if output.exists():
        raise FileExistsError("formal output exists; refusing implicit resume")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / "execution.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        if args.wait_for_audit:
            wait_for_audit(args.wait_for_audit, timeout=10800)
            dependency = json.loads((ROOT / "benchmarks" / f"{args.wait_for_audit}_audit_offset2000.json")
                                    .read_text(encoding="utf-8"))
            if dependency["exact_mean_final"] >= 69:
                print("Dependency reached local 69; defer new training for confirmation.", flush=True)
                return
        if output.exists():
            raise FileExistsError("formal output appeared while waiting")
        baseline = "v230_control" if args.arm == "two" else "v230_eight"
        control_path = ROOT / "benchmarks" / f"{baseline}_audit_offset2000.json"
        control = json.loads(control_path.read_text(encoding="utf-8"))["training_report"]
        paths = [ROOT / config["design"], ROOT / "scripts/train_pure_neural_snr_experts.py", control_path]
        paths += [ROOT / config["parent"] / name for name in ("encoder.pth", "transmitter.pth", "receiver.pth")]
        before = fingerprints(paths)
        record = {**config, "command": command(args.arm), "input_sha256": before,
                  "short_budget_control": baseline, "additional_controlled_steps": 24000}
        (output.parent / "execution_plan.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(json.dumps(record), flush=True)
        subprocess.run(record["command"], cwd=ROOT, check=True)
        report = json.loads((output / "training_report.json").read_text(encoding="utf-8"))
        if report["requested_steps"] != 36000 or report["history"][-1]["step"] != 36000:
            raise RuntimeError("36k training did not complete")
        prefix = check_prefix(report, control)
        (output.parent / "prefix_check.json").write_text(json.dumps(prefix, indent=2), encoding="utf-8")
        if fingerprints(paths) != before or not prefix["matched"]:
            raise RuntimeError("inputs/prefix changed; inspect before treating this as a nested budget ablation")
        subprocess.run([
            sys.executable, "-u", "scripts/audit_pure_neural_candidate.py", "--submission", config["output"],
            "--label", config["label"], "--baseline-label", "v230_eight", "--control-label", "v230_control",
        ], cwd=ROOT, check=True)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()

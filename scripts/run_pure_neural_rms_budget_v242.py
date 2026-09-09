"""Nested RMS-loss 12k -> 36k budgets, retaining the original validation trajectory."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from run_pure_neural_budget_v240 import check_prefix
from run_pure_neural_lr_v237 import ROOT, experiment, fingerprints, train_command

PROTOCOL = {"split_seed": 1176, "test_offset": 2000, "samples": 2000,
            "noise_seeds": [22701, 22702, 22703]}


def plan(arm: str) -> dict:
    result = experiment(arm)
    result.update(label=f"v242_{arm}_rms_36k", learning_rate="1e-5",
                  output=f"artifacts/pure_neural_v242/{arm}/steps36000",
                  short_control="v239_rms_score" if arm == "two" else "v241_eight_rms")
    return result


def command(arm: str) -> list[str]:
    result = train_command(plan(arm))
    result[2] = "scripts/train_pure_neural_rms_v239.py"
    for flag, value in (("--steps", "36000"), ("--patience", "36"), ("--loss-kind", "rms_score")):
        result[result.index(flag) + 1] = value
    return result


def require_audit(report: dict, label: str, steps: int) -> None:
    training = report.get("training_report", {})
    if (report.get("label") != label or report.get("protocol") != PROTOCOL
            or len(report.get("per_seed", [])) != 3 or training.get("requested_steps") != steps
            or not training.get("history") or training["history"][-1]["step"] != steps):
        raise ValueError(f"expected complete {steps}-step fixed audit: {label}")


def wait_eight_slot(timeout: float) -> dict:
    label = "v240_eight_36k"
    path = ROOT / "benchmarks" / f"{label}_audit_offset2000.json"
    deadline = time.monotonic() + timeout
    print(json.dumps({"waiting_for_audit": label, "expected_steps": 36000}), flush=True)
    while True:
        if path.exists():
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
            else:
                require_audit(result, label, 36000)
                return result
        if time.monotonic() >= deadline:
            raise TimeoutError("V240 eight has not completed; inspect instead of duplicating training")
        time.sleep(min(30., max(.01, deadline - time.monotonic())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("two", "eight"), required=True)
    parser.add_argument("--wait-timeout", type=float, default=21600)
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
        if args.arm == "eight" and wait_eight_slot(args.wait_timeout)["exact_mean_final"] >= 69:
            print("V240 reached local 69; defer new training for frozen confirmation.", flush=True)
            return
        if output.exists():
            raise FileExistsError("formal output appeared while waiting")
        control_path = ROOT / "benchmarks" / f"{config['short_control']}_audit_offset2000.json"
        control = json.loads(control_path.read_text(encoding="utf-8"))
        require_audit(control, config["short_control"], 12000)
        if control["training_report"].get("loss_kind") != "rms_score":
            raise ValueError("short-budget control must use RMS loss")
        if control["exact_mean_final"] >= 69:
            print("Short control reached local 69; defer for confirmation.", flush=True)
            return
        paths = [ROOT / config["design"], control_path,
                 ROOT / "scripts/train_pure_neural_snr_experts.py",
                 ROOT / "scripts/train_pure_neural_rms_v239.py",
                 ROOT / "scripts/run_pure_neural_budget_v240.py",
                 ROOT / "scripts/run_pure_neural_lr_v237.py"]
        paths += [ROOT / config["parent"] / name for name in ("encoder.pth", "transmitter.pth", "receiver.pth")]
        before = fingerprints(paths)
        record = {**config, "command": command(args.arm), "input_sha256": before,
                  "additional_controlled_steps": 24000, "objective": "rms_score"}
        (output.parent / "execution_plan.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(json.dumps(record), flush=True)
        subprocess.run(record["command"], cwd=ROOT, check=True)
        report = json.loads((output / "training_report.json").read_text(encoding="utf-8"))
        if (report.get("requested_steps") != 36000 or report["history"][-1]["step"] != 36000
                or report.get("loss_kind") != "rms_score"):
            raise RuntimeError("registered RMS 36k training did not complete")
        prefix = check_prefix(report, control["training_report"])
        (output.parent / "prefix_check.json").write_text(json.dumps(prefix, indent=2), encoding="utf-8")
        if fingerprints(paths) != before or not prefix["matched"]:
            raise RuntimeError("inputs/prefix changed; do not treat as nested budget ablation")
        subprocess.run([
            sys.executable, "-u", "scripts/audit_pure_neural_candidate.py", "--submission", config["output"],
            "--label", config["label"], "--baseline-label", "v240_two_36k",
            "--control-label", config["short_control"],
        ], cwd=ROOT, check=True)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()

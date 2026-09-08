"""Run the preregistered learning-rate x expert-count ablation, then audit it.

Each arm is fresh from V227, never an implicit optimizer resume. Optional audit
dependencies serialize this arm after an existing training slot finishes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def experiment(arm: str) -> dict:
    if arm not in {"two", "eight"}:
        raise ValueError("unknown arm")
    return {
        "arm": arm,
        "label": f"v237_{arm}_lr5",
        "design": f"research/pure_neural_v{227 if arm == 'two' else 230}/modelDesign.py",
        "parent": "artifacts/pure_neural_v227/joint_low",
        "mapping": [0, 1] if arm == "two" else [0, 0, 1, 1, 1, 1, 1, 1],
        "output": f"artifacts/pure_neural_v237/{arm}/lr5",
        "learning_rate": "5e-5",
    }


def train_command(plan: dict) -> list[str]:
    return [
        sys.executable, "-u", "scripts/train_pure_neural_snr_experts.py",
        "--stage", "calibrate", "--model-design", plan["design"],
        "--baseline-dir", plan["parent"], "--baseline-expert-map",
        *map(str, plan["mapping"]), "--output-dir", plan["output"],
        "--train-components", "transmitter", "receiver",
        "--steps", "12000", "--batch-size", "100", "--learning-rate", plan["learning_rate"],
        "--loss-kind", "soft_score", "--score-temperature", "0.5", "--tail-fraction", "0.1",
        "--score-fairness-weight", "0.3", "--quantile-bandwidth", "0.025", "--score-bce-weight", "0.05",
        "--validate-every", "1000", "--validation-samples", "2000", "--patience", "12",
        "--seed", "15240", "--gpu-memory-fraction", "0.4",
    ]


def require_complete(report: dict) -> None:
    if (report.get("requested_steps") != 12000
            or not report.get("history") or report["history"][-1]["step"] != 12000):
        raise ValueError("expected a completed 12000-step training run")


def wait_for_audit(label: str, timeout: float) -> None:
    if not label or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in label):
        raise ValueError("invalid audit dependency label")
    path = ROOT / "benchmarks" / f"{label}_audit_offset2000.json"
    deadline = time.monotonic() + timeout
    print(json.dumps({"waiting_for_audit": label}), flush=True)
    while True:
        if path.exists():
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass  # The previous process may still be flushing the report.
            else:
                if report.get("label") != label or len(report.get("per_seed", [])) != 3:
                    raise ValueError("dependency audit is incomplete or mismatched")
                require_complete(report["training_report"])
                return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"dependency did not complete: {path}")
        time.sleep(min(30.0, max(0.01, deadline - time.monotonic())))


def fingerprints(paths: list[Path]) -> dict[str, str]:
    result = {}
    for path in paths:
        with path.open("rb") as stream:
            result[str(path.relative_to(ROOT))] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("two", "eight"), required=True)
    parser.add_argument("--wait-for-audit")
    parser.add_argument("--wait-timeout", type=float, default=10800)
    args = parser.parse_args()
    plan = experiment(args.arm)
    output = ROOT / plan["output"]
    if output.exists():
        raise FileExistsError("formal output already exists; refusing implicit resume/overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / "execution.lock"
    # An exclusive per-arm lock also prevents duplicate waiting workers.
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        if args.wait_for_audit:
            wait_for_audit(args.wait_for_audit, args.wait_timeout)
        if output.exists():
            raise FileExistsError("output appeared while waiting")
        paths = [ROOT / plan["design"], ROOT / "scripts/train_pure_neural_snr_experts.py"]
        paths += [ROOT / plan["parent"] / name for name in ("encoder.pth", "transmitter.pth", "receiver.pth")]
        before = fingerprints(paths)
        command = train_command(plan)
        record = {**plan, "command": command, "input_sha256": before, "dependency": args.wait_for_audit}
        (output.parent / "execution_plan.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(json.dumps(record), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        require_complete(json.loads((output / "training_report.json").read_text(encoding="utf-8")))
        if fingerprints(paths) != before:
            raise RuntimeError("training inputs changed; do not audit as the registered experiment")
        subprocess.run([
            sys.executable, "-u", "scripts/audit_pure_neural_candidate.py",
            "--submission", plan["output"], "--label", plan["label"],
            "--baseline-label", "v230_eight", "--control-label", "v230_control",
        ], cwd=ROOT, check=True)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()

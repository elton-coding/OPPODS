"""Complete the 12k expert-count x RMS-loss factorial after a 36k slot frees."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from run_pure_neural_lr_v237 import ROOT, fingerprints, require_complete, train_command

PLAN = {
    "arm": "eight_rms", "label": "v241_eight_rms",
    "design": "research/pure_neural_v230/modelDesign.py",
    "parent": "artifacts/pure_neural_v227/joint_low", "mapping": [0, 0, 1, 1, 1, 1, 1, 1],
    "output": "artifacts/pure_neural_v241/eight_rms", "learning_rate": "1e-5",
}
DEPENDENCY = "v240_two_36k"


def command() -> list[str]:
    result = train_command(PLAN)
    result[2] = "scripts/train_pure_neural_rms_v239.py"
    result[result.index("--loss-kind") + 1] = "rms_score"
    return result


def require_dependency(report: dict) -> None:
    training = report.get("training_report", {})
    expected = {"split_seed": 1176, "test_offset": 2000, "samples": 2000,
                "noise_seeds": [22701, 22702, 22703]}
    if (report.get("label") != DEPENDENCY or report.get("protocol") != expected
            or len(report.get("per_seed", [])) != 3 or training.get("requested_steps") != 36000
            or not training.get("history") or training["history"][-1]["step"] != 36000):
        raise ValueError("V241 requires the complete registered 36k two-expert audit")


def wait_dependency(timeout: float) -> dict:
    path = ROOT / "benchmarks" / f"{DEPENDENCY}_audit_offset2000.json"
    deadline = time.monotonic() + timeout
    print(json.dumps({"waiting_for_audit": DEPENDENCY, "expected_steps": 36000}), flush=True)
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
            raise TimeoutError("36k dependency has not completed; inspect its queue")
        time.sleep(min(30.0, max(.01, deadline - time.monotonic())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-timeout", type=float, default=21600)
    args = parser.parse_args()
    output = ROOT / PLAN["output"]
    if output.exists():
        raise FileExistsError("formal output already exists; refusing implicit resume")
    probe = json.loads((ROOT / "artifacts/resource_probe/v241/cpu_adapter/training_report.json")
                       .read_text(encoding="utf-8"))
    if (probe["loss_kind"] != "rms_score" or probe["history"][-1]["step"] != 2
            or probe["baseline_expert_map"] != PLAN["mapping"]):
        raise ValueError("eight-expert RMS CPU training probe must complete first")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / "execution.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        dependency = wait_dependency(args.wait_timeout)
        if dependency["exact_mean_final"] >= 69:
            print("Dependency reached local 69; defer combination for frozen confirmation.", flush=True)
            return
        if output.exists():
            raise FileExistsError("output appeared while waiting")
        paths = [ROOT / PLAN["design"], ROOT / "scripts/train_pure_neural_snr_experts.py",
                 ROOT / "scripts/train_pure_neural_rms_v239.py"]
        paths += [ROOT / PLAN["parent"] / name for name in ("encoder.pth", "transmitter.pth", "receiver.pth")]
        before = fingerprints(paths)
        record = {**PLAN, "command": command(), "input_sha256": before, "dependency": DEPENDENCY}
        (output.parent / "execution_plan.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(json.dumps(record), flush=True)
        subprocess.run(record["command"], cwd=ROOT, check=True)
        report = json.loads((output / "training_report.json").read_text(encoding="utf-8"))
        require_complete(report)
        if report["loss_kind"] != "rms_score" or fingerprints(paths) != before:
            raise RuntimeError("objective or training inputs changed; do not audit")
        subprocess.run([
            sys.executable, "-u", "scripts/audit_pure_neural_candidate.py", "--submission", PLAN["output"],
            "--label", PLAN["label"], "--baseline-label", "v230_eight", "--control-label", "v230_control",
        ], cwd=ROOT, check=True)
        factorial_output = ROOT / "benchmarks/v241_expert_rms_factorial.json"
        if factorial_output.exists():
            raise FileExistsError("factorial report already exists")
        comparison = [sys.executable, "scripts/compare_factorial_holdout.py"]
        for arm, label in (("c", "v230_control"), ("a", "v230_eight"),
                           ("b", "v239_rms_score"), ("ab", PLAN["label"])):
            comparison += [f"--{arm}", *[f"benchmarks/{label}_holdout1176_offset2000_noise{seed}.npz"
                                         for seed in (22701, 22702, 22703)]]
        comparison += ["--factor-a", "eight_experts", "--factor-b", "rms_score",
                       "--output", str(factorial_output)]
        subprocess.run(comparison, cwd=ROOT, check=True)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()

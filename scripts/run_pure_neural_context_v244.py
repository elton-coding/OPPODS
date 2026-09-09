"""Resource-gated zero-initialized joint token-context ablation from V227."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from run_pure_neural_lr_v237 import ROOT, fingerprints, require_complete, train_command
from run_pure_neural_rms_budget_v242 import require_audit

PLAN = {"arm": "token_context", "label": "v244_token_context",
        "design": "research/pure_neural_v244/modelDesign.py",
        "parent": "artifacts/pure_neural_v227/joint_low", "mapping": [0, 1],
        "output": "artifacts/pure_neural_v244/context", "learning_rate": "1e-5"}
DEPENDENCY = "v242_two_rms_36k"


def probe_command(device: str) -> list[str]:
    if device not in {"cpu", "cuda"}:
        raise ValueError("probe device must be cpu or cuda")
    result = train_command(PLAN)
    for flag, value in (("--output-dir", f"artifacts/resource_probe/v244/{device}_adapter"),
                        ("--steps", "2"), ("--validate-every", "2"), ("--validation-samples", "4"),
                        ("--batch-size", "2" if device == "cpu" else "100")):
        result[result.index(flag) + 1] = value
    if device == "cpu":
        index = result.index("--gpu-memory-fraction")
        del result[index:index + 2]
    result += ["--device", device]
    return result


def wait_slot(timeout: float) -> dict:
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
                require_audit(report, DEPENDENCY, 36000)
                return report
        if time.monotonic() >= deadline:
            raise TimeoutError("V242 two dependency incomplete; inspect its existing process")
        time.sleep(min(30., max(.01, deadline - time.monotonic())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-timeout", type=float, default=21600)
    args = parser.parse_args()
    output = ROOT / PLAN["output"]
    probe_output = ROOT / "artifacts/resource_probe/v244/cuda_adapter"
    if output.exists() or probe_output.exists():
        raise FileExistsError("formal/GPU probe output exists; refusing implicit resume")
    cpu = json.loads((ROOT / "artifacts/resource_probe/v244/cpu_adapter/training_report.json")
                     .read_text(encoding="utf-8"))
    if cpu["history"][-1]["step"] != 2 or cpu["baseline_expert_map"] != [0, 1]:
        raise ValueError("complete CPU parent-adapter probe required")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / "execution.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        dependency = wait_slot(args.wait_timeout)
        if dependency["exact_mean_final"] >= 69:
            print("Dependency reached local 69; defer new architecture for frozen confirmation.", flush=True)
            return
        if output.exists() or probe_output.exists():
            raise FileExistsError("formal/GPU probe appeared while waiting")
        paths = [ROOT / PLAN["design"], ROOT / "scripts/train_pure_neural_snr_experts.py"]
        paths += [ROOT / PLAN["parent"] / name for name in ("encoder.pth", "transmitter.pth", "receiver.pth")]
        before = fingerprints(paths)
        record = {**PLAN, "input_sha256": before, "dependency": DEPENDENCY,
                  "probe_command": probe_command("cuda"), "command": train_command(PLAN)}
        (output.parent / "execution_plan.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(json.dumps(record), flush=True)
        subprocess.run(record["probe_command"], cwd=ROOT, check=True)
        probe = json.loads((probe_output / "training_report.json").read_text(encoding="utf-8"))
        if (probe["history"][-1]["step"] != 2 or probe["batch_size"] != 100
                or probe["gpu_peak_allocated_bytes"] <= 0):
            raise RuntimeError("full-batch GPU resource probe did not complete")
        if fingerprints(paths) != before or output.exists():
            raise RuntimeError("inputs changed or formal output appeared")
        subprocess.run(record["command"], cwd=ROOT, check=True)
        require_complete(json.loads((output / "training_report.json").read_text(encoding="utf-8")))
        if fingerprints(paths) != before:
            raise RuntimeError("training inputs changed; do not audit")
        subprocess.run([
            sys.executable, "-u", "scripts/audit_pure_neural_candidate.py", "--submission", PLAN["output"],
            "--label", PLAN["label"], "--baseline-label", "v240_two_36k", "--control-label", "v230_control",
        ], cwd=ROOT, check=True)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()

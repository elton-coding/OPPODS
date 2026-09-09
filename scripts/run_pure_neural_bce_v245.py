"""Sequential, single-factor BCE auxiliary-weight ablations from the fixed V227 parent."""
from __future__ import annotations

import json
import os
import subprocess
import sys

from run_pure_neural_lr_v237 import ROOT, fingerprints, require_complete, train_command

ARMS = {"zero": "0", "low": "0.01"}


def plan(arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError("unknown BCE arm")
    return {"arm": arm, "label": f"v245_bce_{arm}", "bce_weight": ARMS[arm],
            "design": "research/pure_neural_v227/modelDesign.py",
            "parent": "artifacts/pure_neural_v227/joint_low", "mapping": [0, 1],
            "output": f"artifacts/pure_neural_v245/{arm}", "learning_rate": "1e-5"}


def command(arm: str, *, probe: bool = False) -> list[str]:
    config = plan(arm)
    result = train_command(config)
    result[result.index("--score-bce-weight") + 1] = config["bce_weight"]
    if probe:
        for flag, value in (("--output-dir", f"artifacts/resource_probe/v245/{arm}"),
                            ("--steps", "2"), ("--validate-every", "2"),
                            ("--validation-samples", "4")):
            result[result.index(flag) + 1] = value
    return result


def require_arm(report: dict, arm: str, *, probe: bool = False) -> None:
    if not probe:
        require_complete(report)
    expected_steps = 2 if probe else 12000
    if (report.get("requested_steps") != expected_steps
            or not report.get("history") or report["history"][-1]["step"] != expected_steps
            or report.get("score_bce_weight") != float(ARMS[arm])
            or report.get("loss_kind") != "soft_score"
            or report.get("batch_size") != 100
            or report.get("baseline_expert_map") != [0, 1]
            or report.get("gpu_peak_allocated_bytes", 0) <= 0):
        raise ValueError("incomplete or mismatched BCE arm")


def main() -> None:
    root = ROOT / "artifacts/pure_neural_v245"
    outputs = [ROOT / plan(arm)["output"] for arm in ARMS]
    probes = [ROOT / f"artifacts/resource_probe/v245/{arm}" for arm in ARMS]
    if any(path.exists() for path in outputs + probes):
        raise FileExistsError("an output exists; inspect evidence instead of implicitly resuming")
    # V244 is the slot predecessor; this runner never launches two arms together.
    dependency = json.loads((ROOT / "benchmarks/v244_token_context_audit_offset2000.json")
                            .read_text(encoding="utf-8"))
    require_complete(dependency["training_report"])
    if dependency.get("label") != "v244_token_context" or len(dependency.get("per_seed", [])) != 3:
        raise ValueError("V244 audit must be complete before taking its slot")
    if dependency["exact_mean_final"] >= 69:
        print("Defer new training: freeze and confirm V244 first.", flush=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "execution.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        for arm in ARMS:
            config = plan(arm)
            paths = [ROOT / config["design"], ROOT / "scripts/train_pure_neural_snr_experts.py",
                     ROOT / "scripts/run_pure_neural_lr_v237.py", ROOT / "scripts/run_pure_neural_bce_v245.py"]
            paths += [ROOT / config["parent"] / name for name in
                      ("encoder.pth", "transmitter.pth", "receiver.pth")]
            before = fingerprints(paths)
            record = {**config, "input_sha256": before, "command": command(arm),
                      "probe_command": command(arm, probe=True), "dependency": "v244_token_context"}
            with (root / f"{arm}_execution_plan.json").open("x", encoding="utf-8") as stream:
                json.dump(record, stream, indent=2)
            print(json.dumps(record), flush=True)
            subprocess.run(record["probe_command"], cwd=ROOT, check=True)
            probe_path = ROOT / f"artifacts/resource_probe/v245/{arm}/training_report.json"
            probe_report = json.loads(probe_path.read_text(encoding="utf-8"))
            require_arm(probe_report, arm, probe=True)
            with (ROOT / f"benchmarks/v245_{arm}_gpu_training_probe.json").open("x", encoding="utf-8") as stream:
                json.dump({"purpose": "runtime check only, not performance evidence", "report": probe_report},
                          stream, indent=2)
            if fingerprints(paths) != before or (ROOT / config["output"]).exists():
                raise RuntimeError("inputs changed or formal output appeared")
            subprocess.run(record["command"], cwd=ROOT, check=True)
            report = json.loads((ROOT / config["output"] / "training_report.json").read_text(encoding="utf-8"))
            require_arm(report, arm)
            if fingerprints(paths) != before:
                raise RuntimeError("training inputs changed; do not audit")
            subprocess.run([sys.executable, "-u", "scripts/audit_pure_neural_candidate.py",
                            "--submission", config["output"], "--label", config["label"],
                            "--baseline-label", "v240_two_36k", "--control-label", "v230_control"],
                           cwd=ROOT, check=True)
            audit = json.loads((ROOT / f"benchmarks/{config['label']}_audit_offset2000.json")
                               .read_text(encoding="utf-8"))
            if audit["exact_mean_final"] >= 69:
                print("Local 69 reached; remaining arm deferred for frozen confirmation.", flush=True)
                break
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()

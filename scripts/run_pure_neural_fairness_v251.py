"""V251: isolate surrogate fairness weight .3 -> .5 at eight-expert RMS 36k."""
from __future__ import annotations

import json
import os
import subprocess
import sys

from audit_pure_neural_candidate import fingerprint
from run_pure_neural_long_budget_v250 import CONTROL, data_record, require_report
from run_pure_neural_lr_v237 import ROOT, fingerprints
from run_pure_neural_rms_budget_v242 import command as control_command
from run_pure_neural_rms_budget_v242 import require_audit

LABEL = "v251_eight_rms_fair50_36k"
OUTPUT = "artifacts/pure_neural_v251/eight/fair50_steps36000"


def command():
    result = control_command("eight")
    for flag, value in (("--score-fairness-weight", "0.5"), ("--output-dir", OUTPUT)):
        result[result.index(flag) + 1] = value
    return result


def require_candidate(report):
    if report.get("score_fairness_weight") != .5:
        raise ValueError("V251 requires surrogate fairness .5")
    # All other protocol fields are exactly the 36k control; do not mutate the report.
    require_report({**report, "score_fairness_weight": .3}, 36000)


def check_initial(report, control):
    differences = {key: abs(report["history"][0][key] - control["history"][0][key])
                   for key in ("loss", "efficiency", "fairness", "final")}
    if max(differences.values()) > 1e-4:
        raise ValueError("initial validation differs from original eight-expert control")
    return differences


def main():
    output = ROOT / OUTPUT
    if output.exists():
        raise FileExistsError("formal output exists; refusing implicit resume")
    control_path = ROOT / f"benchmarks/{CONTROL}_audit_offset2000.json"
    control = json.loads(control_path.read_text(encoding="utf-8"))
    require_audit(control, CONTROL, 36000)
    require_report(control["training_report"], 36000)
    if fingerprint(ROOT / "artifacts/pure_neural_v242/eight/steps36000") != control["files"]:
        raise RuntimeError("frozen 36k control changed")
    if control["exact_mean_final"] >= 69:
        print("Control reached local 69; defer this trial for frozen confirmation.", flush=True)
        return
    original_path = ROOT / "artifacts/pure_neural_v242/eight/execution_plan.json"
    original = json.loads(original_path.read_text(encoding="utf-8"))
    paths = [ROOT / name for name in original["input_sha256"]]
    if fingerprints(paths) != original["input_sha256"]:
        raise RuntimeError("original control training inputs changed")
    paths += [control_path, original_path, *[ROOT / name for name in (
        "ziliao/data_train/H_train.npz", "src/oppods/data.py",
        "scripts/run_pure_neural_rms_budget_v242.py", "scripts/run_pure_neural_long_budget_v250.py",
        "scripts/run_pure_neural_fairness_v251.py", "scripts/audit_pure_neural_candidate.py",
        "scripts/evaluate_submission.py", "scripts/compare_paired_holdout.py")]]
    before, dataset = fingerprints(paths), data_record()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / "execution.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        record = {"label": LABEL, "control": CONTROL, "command": command(), "input_sha256": before,
                  "dataset": dataset, "train_channel_draws": 3600000,
                  "single_factor": "surrogate fairness weight .3 -> .5; official evaluation remains .3",
                  "initialization": "original V227, fresh optimizer, not V242E best weights",
                  "resource_gate": "unchanged V242E architecture, batch100 and RMS entry; completed full-GPU control"}
        with (output.parent / "execution_plan.json").open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
        print(json.dumps(record), flush=True)
        subprocess.run(record["command"], cwd=ROOT, check=True)
        report = json.loads((output / "training_report.json").read_text(encoding="utf-8"))
        require_candidate(report)
        initial = check_initial(report, control["training_report"])
        if fingerprints(paths) != before or data_record() != dataset:
            raise RuntimeError("registered inputs changed; do not audit as a controlled trial")
        with (ROOT / "benchmarks/v251_initial_validation_check.json").open("x", encoding="utf-8") as stream:
            json.dump({"maximum_absolute_differences": initial, "matched": True,
                       "scope": "initial validation only; later trajectories should differ"}, stream, indent=2)
        subprocess.run([sys.executable, "-u", "scripts/audit_pure_neural_candidate.py", "--submission", OUTPUT,
                        "--label", LABEL, "--baseline-label", CONTROL], cwd=ROOT, check=True)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()

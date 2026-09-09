"""V250: extend the verified eight-expert RMS trajectory from 36k to 72k."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import torch
from audit_pure_neural_candidate import fingerprint
from run_pure_neural_lr_v237 import ROOT, fingerprints
from run_pure_neural_rms_budget_v242 import command as short_command
from run_pure_neural_rms_budget_v242 import require_audit

from oppods.data import ChannelMemmap, deterministic_split_indices

LABEL = "v250_eight_rms_72k"
CONTROL = "v242_eight_rms_36k"
OUTPUT = "artifacts/pure_neural_v250/eight/steps72000"
DATA = "ziliao/data_train/H_train.npz"


def command():
    result = short_command("eight")
    for flag, value in (("--steps", "72000"), ("--patience", "72"), ("--output-dir", OUTPUT)):
        result[result.index(flag) + 1] = value
    return result


def require_report(report, steps):
    expected = {"requested_steps": steps, "batch_size": 100, "seed": 15240, "learning_rate": 1e-5,
                "loss_kind": "rms_score", "score_temperature": .5, "score_bce_weight": .05,
                "score_fairness_weight": .3, "quantile_bandwidth": .025,
                "train_components": ["transmitter", "receiver"], "validation_samples": 2000,
                "baseline_expert_map": [0, 0, 1, 1, 1, 1, 1, 1], "gpu_memory_fraction": .4}
    if (any(report.get(key) != value for key, value in expected.items())
            or [row["step"] for row in report.get("history", [])] != list(range(0, steps + 1, 1000))
            or report.get("gpu_peak_allocated_bytes", 0) <= 0):
        raise ValueError("incomplete or mismatched eight-expert RMS budget")


def check_prefix(report, control):
    expected = list(range(0, 36001, 1000))
    prefix = [row for row in report["history"] if row["step"] <= 36000]
    if ([row["step"] for row in prefix] != expected
            or [row["step"] for row in control["history"]] != expected):
        raise ValueError("all 37 validation checkpoints through 36000 are required")
    differences = {metric: max(abs(float(a[metric]) - float(b[metric]))
                               for a, b in zip(prefix, control["history"], strict=True))
                   for metric in ("loss", "efficiency", "fairness", "final")}
    return {"matched": differences["loss"] <= 1e-5 and all(differences[k] <= 1e-4
            for k in ("efficiency", "fairness", "final")), "maximum_absolute_difference": differences,
            "checkpoints": 37, "scope": "same 0..36000 validation trajectory; not proof of bitwise optimizer state equality"}


def data_record():
    data = ChannelMemmap(ROOT / DATA)
    if len(data) != 100000:
        raise ValueError("unexpected channel dataset size")
    split = deterministic_split_indices(len(data), seed=1176)
    return {"path": DATA, "bytes": (ROOT / DATA).stat().st_size, "shape": list(data.shape),
            "storage_dtype": str(data.storage_dtype), "split_seed": 1176,
            "indices": {name: {"count": len(values), "sha256": hashlib.sha256(
                values.astype("<i8").tobytes()).hexdigest()} for name, values in split.items()},
            "validation_first2000_sha256": hashlib.sha256(split["validation"][:2000].astype("<i8").tobytes()).hexdigest(),
            "historical_ancestor_provenance_certified": False}


def main():
    output = ROOT / OUTPUT
    if output.exists():
        raise FileExistsError("formal output exists; refusing implicit optimizer restart")
    control_path = ROOT / f"benchmarks/{CONTROL}_audit_offset2000.json"
    control = json.loads(control_path.read_text(encoding="utf-8"))
    require_audit(control, CONTROL, 36000)
    require_report(control["training_report"], 36000)
    if fingerprint(ROOT / "artifacts/pure_neural_v242/eight/steps36000") != control["files"]:
        raise RuntimeError("36k control changed since audit")
    if control["exact_mean_final"] >= 69:
        print("Control reached local 69; defer extra training for frozen confirmation.", flush=True)
        return
    original_plan_path = ROOT / "artifacts/pure_neural_v242/eight/execution_plan.json"
    original = json.loads(original_plan_path.read_text(encoding="utf-8"))
    paths = [ROOT / name for name in original["input_sha256"]]
    if fingerprints(paths) != original["input_sha256"]:
        raise RuntimeError("original V242 training inputs changed; prefix replication not registered")
    paths += [control_path, original_plan_path, ROOT / DATA, ROOT / "src/oppods/data.py",
              ROOT / "scripts/run_pure_neural_rms_budget_v242.py", ROOT / "scripts/run_pure_neural_long_budget_v250.py"]
    before = fingerprints(paths)
    dataset = data_record()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / "execution.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        record = {"label": LABEL, "control": CONTROL, "command": command(), "input_sha256": before,
                  "dataset": dataset, "train_channel_draws": 7200000, "additional_steps": 36000,
                  "optimizer_restart_from_36k_checkpoint": False,
                  "environment": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__},
                  "resource_gate": "same full batch100 architecture and entry as completed V242; no new GPU probe needed"}
        with (output.parent / "execution_plan.json").open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
        print(json.dumps(record), flush=True)
        subprocess.run(record["command"], cwd=ROOT, check=True)
        report = json.loads((output / "training_report.json").read_text(encoding="utf-8"))
        require_report(report, 72000)
        prefix = check_prefix(report, control["training_report"])
        with (ROOT / "benchmarks/v250_eight_rms_72k_prefix_check.json").open("x", encoding="utf-8") as stream:
            json.dump(prefix, stream, indent=2)
        if not prefix["matched"] or fingerprints(paths) != before or data_record() != dataset:
            raise RuntimeError("inputs or 36k prefix differ; do not audit as registered budget ablation")
        subprocess.run([sys.executable, "-u", "scripts/audit_pure_neural_candidate.py", "--submission", OUTPUT,
                        "--label", LABEL, "--baseline-label", CONTROL, "--control-label", "v241_eight_rms"],
                       cwd=ROOT, check=True)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()

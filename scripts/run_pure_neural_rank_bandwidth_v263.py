"""V263: only widen the V257 RMS P10 rank kernel .025 -> .05; fresh 72k."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys

from audit_pure_neural_candidate import MODEL_FILES, fingerprint, score_paths
from compare_paired_holdout import compare
from run_frozen_confirmation_v258 import checked_caches
from run_pure_neural_fairness_v251 import check_initial
from run_pure_neural_long_budget_v250 import data_record
from run_pure_neural_lr_v237 import ROOT, fingerprints
from run_pure_neural_rms_budget_v242 import require_audit
from run_pure_neural_shared_v257 import LABEL as CONTROL
from run_pure_neural_shared_v257 import OUTPUT as CONTROL_DIR
from run_pure_neural_shared_v257 import command as control_command
from run_pure_neural_shared_v257 import require_result as control_guard
from run_storage_factorial_v254 import verify_bound_inputs, wait_gpu_slot, write_new

LABEL = "v263_shared8_rms_bandwidth050_72k"
OUTPUT = "artifacts/pure_neural_v263/eight/bandwidth050_steps72000"
PROBE = "artifacts/resource_probe/v263/eight_bandwidth050"
BASE = ROOT / "artifacts/pure_neural_v263/eight"
PLAN = BASE / "execution_plan.json"
DECISION = ROOT / "benchmarks/v263_selection_decision.json"
PROBE_REPORT = ROOT / "benchmarks/v263_gpu_training_probe.json"
INITIAL = ROOT / "benchmarks/v263_initial_validation_check.json"
SEEDS = [22701, 22702, 22703]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def command(*, probe=False):
    result = control_command(probe=probe)
    result[result.index("--quantile-bandwidth") + 1] = "0.05"
    result[result.index("--output-dir") + 1] = PROBE if probe else OUTPUT
    return result


def require_result(report, *, probe=False):
    if report.get("quantile_bandwidth") != .05:
        raise ValueError("V263 requires rank bandwidth .05")
    if any(name in report for name in ("encoder_learning_rate", "microbatch_size", "learning_rate_schedule")):
        raise ValueError("no additional training factors permitted")
    control_guard({**report, "quantile_bandwidth": .025}, probe=probe)
    for row in report["history"]:
        if not all(math.isfinite(row[k]) for k in ("loss", "efficiency", "fairness", "final")):
            raise ValueError("nonfinite validation metrics")
    if report.get("best_step") not in [row["step"] for row in report["history"]]:
        raise ValueError("best checkpoint absent from complete history")


def decision(comparison):
    deltas = [row["delta"]["final"] for row in comparison["per_seed"]]
    interval = comparison["paired_delta_95_percentile_interval"]
    if ([row["noise_seed"] for row in comparison["per_seed"]] != SEEDS
            or len(interval) != 2 or not all(math.isfinite(v) for v in [*deltas, *interval])
            or interval[0] > interval[1]):
        raise ValueError("invalid paired statistics")
    return {"comparison": comparison,
            "eligible_for_new_confirmation": len(deltas) == 3 and all(v > 0 for v in deltas) and interval[0] > 0,
            "automatic_promotion": False, "online_confirmation": False,
            "historical_test_reuse": True,
            "caveat": "selection evidence only; fresh confirmation provenance must be checked separately"}


def require_metrics(audit, cached):
    rows = audit["per_seed"]
    if ([row.get("seed") for row in rows] != SEEDS
            or any(row.get("samples") != 2000 or row.get("scores") != 4000
                   or row.get("short_outputs") != 0 or row.get("min_output_length") != 1152
                   or row.get("max_output_length") != 1152 for row in rows)):
        raise ValueError("wrong full-payload audit protocol")
    for row, actual in zip(rows, cached, strict=True):
        for key, mapped in (("final", "final"), ("efficiency", "efficiency"), ("fairness", "p10")):
            if not math.isfinite(row[key]) or abs(row[key] - actual[mapped]) > 1e-4:
                raise ValueError("audit metrics do not reproduce from cache")
    mean = sum(row["final"] for row in rows) / 3
    if not math.isfinite(audit["exact_mean_final"]) or abs(mean - audit["exact_mean_final"]) > 1e-10:
        raise ValueError("audit mean mismatch")


def main():
    audit_path = ROOT / f"benchmarks/{LABEL}_audit_offset2000.json"
    reserved = [ROOT / OUTPUT, ROOT / PROBE, PLAN, BASE / "execution.lock", BASE / "failure.json",
                DECISION, PROBE_REPORT, INITIAL, audit_path,
                *score_paths(ROOT / "benchmarks", LABEL, SEEDS, 2000)]
    if any(path.exists() for path in reserved):
        raise FileExistsError("V263 already has outputs; no automatic restart/resume/overwrite")
    original_path = ROOT / "artifacts/pure_neural_v262/eight/execution_plan.json"
    original = read(original_path)
    verify_bound_inputs(original)
    control_path = ROOT / f"benchmarks/{CONTROL}_audit_offset2000.json"
    control = read(control_path)
    require_audit(control, CONTROL, 72000)
    control_guard(control["training_report"])
    if (control["files"] != fingerprint(ROOT / CONTROL_DIR)
            or control["training_report"] != read(ROOT / CONTROL_DIR / "training_report.json")):
        raise ValueError("frozen V257 control changed")
    require_metrics(control, checked_caches([CONTROL], offset=2000, seeds=SEEDS)[CONTROL])
    paths = [ROOT / name for name in original["input_sha256"]]
    paths += [original_path, control_path, ROOT / CONTROL_DIR / "training_report.json"]
    paths += [ROOT / CONTROL_DIR / name for name in MODEL_FILES]
    paths += score_paths(ROOT / "benchmarks", CONTROL, SEEDS, 2000)
    paths += [ROOT / name for name in (
        "scripts/run_pure_neural_rank_bandwidth_v263.py", "scripts/run_frozen_confirmation_v258.py",
        "tests/test_rank_bandwidth_v263.py", "docs/experiments/pure-neural-rank-bandwidth-v263.md")]
    plan = {"label": LABEL, "control": CONTROL, "command": command(), "probe_command": command(probe=True),
            "input_sha256": fingerprints(paths), "dataset": data_record(), "train_channel_draws": 7200000,
            "single_factor": "Gaussian P10 rank bandwidth .025 -> .05; V257 shared-prefix architecture unchanged",
            "initialization": "fresh original V227 mapping [0,0,1,1,1,1,1,1], fresh Adam; no probe or champion reuse",
            "criterion": "all 3 paired final deltas positive AND bootstrap95 lower bound positive",
            "bootstrap_seed": 227, "bootstrap_repeats": 2000, "automatic_promotion": False}
    BASE.mkdir(parents=True, exist_ok=True)
    lock = BASE / "execution.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        write_new(PLAN, plan)
        wait_gpu_slot(timeout=86400)
        subprocess.run(plan["probe_command"], cwd=ROOT, check=True)
        probe = read(ROOT / PROBE / "training_report.json")
        require_result(probe, probe=True)
        initial = check_initial(probe, control["training_report"])
        if fingerprint(ROOT / PROBE)["modelDesign.py"] != fingerprint(ROOT / CONTROL_DIR)["modelDesign.py"]:
            raise ValueError("probe architecture changed")
        write_new(PROBE_REPORT, {"report": probe, "initial_differences": initial,
                                 "purpose": "full batch100 runtime only, not score evidence"})
        verify_bound_inputs(plan)
        wait_gpu_slot(timeout=86400)
        subprocess.run(plan["command"], cwd=ROOT, check=True)
        report = read(ROOT / OUTPUT / "training_report.json")
        require_result(report)
        initial = check_initial(report, control["training_report"])
        if fingerprint(ROOT / OUTPUT)["modelDesign.py"] != control["files"]["modelDesign.py"]:
            raise ValueError("formal model design changed")
        verify_bound_inputs(plan)
        if data_record() != plan["dataset"]:
            raise ValueError("data record changed")
        write_new(INITIAL, {"matched": True, "maximum_absolute_differences": initial})
        wait_gpu_slot(timeout=86400)
        subprocess.run([sys.executable, "-u", "scripts/audit_pure_neural_candidate.py", "--submission", OUTPUT,
                        "--label", LABEL, "--baseline-label", CONTROL], cwd=ROOT, check=True)
        audit = read(audit_path)
        require_audit(audit, LABEL, 72000)
        if audit["training_report"] != report or audit["files"] != fingerprint(ROOT / OUTPUT):
            raise ValueError("audit source mismatch")
        caches = checked_caches([CONTROL, LABEL], offset=2000, seeds=SEEDS)
        require_metrics(control, caches[CONTROL])
        require_metrics(audit, caches[LABEL])
        comparison = compare(score_paths(ROOT / "benchmarks", CONTROL, SEEDS, 2000),
                             score_paths(ROOT / "benchmarks", LABEL, SEEDS, 2000))
        if comparison != audit["comparisons"][CONTROL]:
            raise ValueError("paired comparison does not reproduce")
        result = decision(comparison)
        result["exact_mean_final"] = audit["exact_mean_final"]
        verify_bound_inputs(plan)
        write_new(DECISION, result)
        print({"mean": audit["exact_mean_final"], "eligible": result["eligible_for_new_confirmation"]}, flush=True)
    except Exception as error:
        write_new(BASE / "failure.json", {"error": repr(error), "partial_outputs_preserved": True,
                                         "implicit_resume_allowed": False})
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()

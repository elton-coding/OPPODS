"""V259: global RMS batch400 at the V250 matched 7.2M-draw budget."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

from audit_pure_neural_candidate import fingerprint, score_paths
from probe_pure_neural_rms_global_v259 import CPU_PROOF, GPU_PROOF, source_paths
from run_pure_neural_fairness_v251 import check_initial
from run_pure_neural_long_budget_v250 import command as control_command
from run_pure_neural_long_budget_v250 import data_record, require_report
from run_pure_neural_lr_v237 import ROOT, fingerprints
from run_pure_neural_rms_budget_v242 import require_audit
from run_pure_neural_shared_v257 import require_result as require_shared_result
from run_storage_factorial_v254 import verify_bound_inputs, wait_gpu_slot, write_new

CONTROL = "v250_eight_rms_72k"
DEPENDENCY = "v257_eight_shared_prefix8_rms_72k"
DEPENDENCY_DIR = "artifacts/pure_neural_v257/eight/shared8_steps72000"
LABEL = "v259_eight_rms_global400_18k"
OUTPUT = "artifacts/pure_neural_v259/eight/global400_steps18000"
PROBE = "artifacts/resource_probe/v259/eight_global400"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def command(*, probe=False):
    result = control_command()
    result[2] = "scripts/train_pure_neural_rms_global_v259.py"
    for flag, value in (("--batch-size", "400"), ("--steps", "2" if probe else "18000"),
                        ("--validate-every", "2" if probe else "250"),
                        ("--output-dir", PROBE if probe else OUTPUT)):
        result[result.index(flag) + 1] = value
    return result + ["--microbatch-size", "100"]


def require_result(report, *, probe=False):
    steps = 2 if probe else 18000
    expected = {"requested_steps": steps, "batch_size": 400, "seed": 15240, "learning_rate": 1e-5,
                "loss_kind": "rms_score", "score_temperature": .5, "score_bce_weight": .05,
                "score_fairness_weight": .3, "quantile_bandwidth": .025,
                "train_components": ["transmitter", "receiver"], "trainable_parameters": 189551584,
                "validation_samples": 2000, "validation_batch_size": 100, "microbatch_size": 100,
                "global_loss_ue_count": 800, "training_channel_draws": steps * 400,
                "baseline_expert_map": [0, 0, 1, 1, 1, 1, 1, 1], "gpu_memory_fraction": .4,
                "variable_payload": False, "shared_frontend": False,
                "activation_checkpointing": "nonreentrant; explicit per-chunk generator replay"}
    history = [0, 2] if probe else list(range(0, 18001, 250))
    if (any(report.get(k) != v for k, v in expected.items())
            or [r["step"] for r in report.get("history", [])] != history
            or report.get("gpu_peak_allocated_bytes", 0) <= 0
            or any(k in report for k in ("learning_rate_schedule", "encoder_learning_rate"))):
        raise ValueError("incomplete/mismatched global RMS batch400 sample-budget report")


def require_runtime(record, device):
    cuda = device == "cuda"
    batch, micro = (400, 100) if cuda else (16, 8)
    expected = {"passed": True, "device": device, "batch_size": batch, "microbatch_size": micro,
                "route_counts": [batch // 8] * 8, "trainable_parameters": 189551584,
                "encoder_unchanged": True, "weights_saved": False, "gpu_memory_fraction": .4 if cuda else None}
    rows = record.get("steps", [])
    if (any(record.get(k) != v for k, v in expected.items())
            or [r["step"] for r in rows] != [1, 2]
            or any(r.get("adam_parameter_states") != 1296 or r.get("gradient_tensors") != 1296
                   or r.get("global_loss_ue_count") != 2 * batch
                   or r.get("backward_preserves_noise_rng") is not True
                   or not math.isfinite(r.get("loss", float("nan"))) for r in rows)):
        raise ValueError("incomplete global loss/checkpoint runtime proof")
    if cuda:
        if record.get("gpu_peak_allocated_bytes", 0) <= 0:
            raise ValueError("missing CUDA resource evidence")
    else:
        identity = record.get("identity") or {}
        if (identity.get("logit_maximum_difference") != 0 or identity.get("loss_difference") != 0
                or identity.get("forward_noise_rng_equal") is not True
                or identity.get("backward_preserves_noise_rng") is not True
                or not math.isfinite(identity.get("maximum_gradient_difference", float("inf")))
                or identity.get("maximum_gradient_difference", float("inf")) > 1e-7):
            raise ValueError("CPU global RMS recomputation identity failed")
    verify_bound_inputs(record)


def wait_dependency(timeout):
    path = ROOT / f"benchmarks/{DEPENDENCY}_audit_offset2000.json"
    deadline = time.monotonic() + timeout
    print(json.dumps({"waiting_for_full_audit": DEPENDENCY, "steps": 72000}), flush=True)
    while True:
        if path.exists():
            try:
                audit = read(path)
            except json.JSONDecodeError:
                pass
            else:
                require_audit(audit, DEPENDENCY, 72000)
                require_shared_result(audit["training_report"])
                if (fingerprint(ROOT / DEPENDENCY_DIR) != audit["files"]
                        or read(ROOT / DEPENDENCY_DIR / "training_report.json") != audit["training_report"]):
                    raise RuntimeError("completed V257 dependency changed")
                paths = [path, ROOT / DEPENDENCY_DIR / "training_report.json"]
                paths += [ROOT / DEPENDENCY_DIR / name for name in audit["files"]]
                paths += score_paths(ROOT / "benchmarks", DEPENDENCY, [22701, 22702, 22703], 2000)
                return {"input_sha256": fingerprints(paths), "exact_mean_final": audit["exact_mean_final"],
                        "resource_dependency_only": True, "v259_control": CONTROL}
        if time.monotonic() >= deadline:
            raise TimeoutError("V257 incomplete; inspect original process, no restart")
        time.sleep(min(30., max(.01, deadline - time.monotonic())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-timeout", type=float, default=86400)
    args = parser.parse_args()
    output, probe = ROOT / OUTPUT, ROOT / PROBE
    reserved = [output, probe, output.parent / "execution_plan.json", output.parent / "execution.lock",
                output.parent / "failure.json", output.parent / "completed_dependency.json", ROOT / GPU_PROOF,
                ROOT / "benchmarks/v259_gpu_training_probe.json", ROOT / "benchmarks/v259_initial_validation_check.json",
                ROOT / f"benchmarks/{LABEL}_audit_offset2000.json"]
    reserved += score_paths(ROOT / "benchmarks", LABEL, [22701, 22702, 22703], 2000)
    if any(p.exists() for p in reserved):
        raise FileExistsError("V259 output exists; no implicit restart/resume/overwrite")
    require_runtime(read(ROOT / CPU_PROOF), "cpu")
    control_path = ROOT / f"benchmarks/{CONTROL}_audit_offset2000.json"
    control = read(control_path)
    require_audit(control, CONTROL, 72000)
    require_report(control["training_report"], 72000)
    if fingerprint(ROOT / "artifacts/pure_neural_v250/eight/steps72000") != control["files"]:
        raise RuntimeError("frozen control changed")
    prior_path = ROOT / "artifacts/pure_neural_v258/eight/execution_plan.json"
    prior = read(prior_path)
    verify_bound_inputs(prior)
    paths = source_paths() + [control_path, ROOT / CPU_PROOF, prior_path]
    paths += [ROOT / n for n in prior["input_sha256"]]
    paths += [ROOT / "scripts/run_pure_neural_rms_global_v259.py"]
    plan = {"label": LABEL, "control": CONTROL, "dependency": DEPENDENCY,
            "input_sha256": fingerprints(paths), "dataset": data_record(), "wait_timeout": args.wait_timeout,
            "command": command(), "probe_command": command(probe=True), "train_channel_draws": 7200000,
            "single_factor": "training grouping batch100 to400 at fixed7.2M draws; same LR, eight-expert RMS",
            "updates": 18000, "microbatch": 100, "global_loss_ue_count": 800,
            "validation_interval_channel_draws": 100000, "validation_batch_size": 100,
            "initialization": "original V227; fresh Adam, not probe/control/dependency checkpoints",
            "caveat": "not equal update count, training random stream or wall-clock compute", "automatic_promotion": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / "execution.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        write_new(output.parent / "execution_plan.json", plan)
        print(json.dumps({"plan": str(output.parent / "execution_plan.json"),
                          "frozen_inputs": len(plan["input_sha256"])}), flush=True)
        dependency = wait_dependency(args.wait_timeout)
        write_new(output.parent / "completed_dependency.json", dependency)
        if max(control["exact_mean_final"], dependency["exact_mean_final"]) >= 69:
            print("Local69 reached; defer new training for confirmation/compliance.", flush=True)
            return
        verify_bound_inputs(plan)
        if data_record() != plan["dataset"]:
            raise RuntimeError("data changed while waiting")
        wait_gpu_slot()
        subprocess.run([sys.executable, "-u", "scripts/probe_pure_neural_rms_global_v259.py", "--device", "cuda"],
                       cwd=ROOT, check=True)
        require_runtime(read(ROOT / GPU_PROOF), "cuda")
        wait_gpu_slot()
        subprocess.run(plan["probe_command"], cwd=ROOT, check=True)
        report = read(probe / "training_report.json")
        require_result(report, probe=True)
        initial = check_initial(report, control["training_report"])
        write_new(ROOT / "benchmarks/v259_gpu_training_probe.json", {"purpose": "initial/runtime only, not score",
                  "report": report, "initial_validation_differences": initial})
        verify_bound_inputs(plan)
        wait_gpu_slot()
        subprocess.run(plan["command"], cwd=ROOT, check=True)
        report = read(output / "training_report.json")
        require_result(report)
        initial = check_initial(report, control["training_report"])
        verify_bound_inputs(plan)
        if data_record() != plan["dataset"]:
            raise RuntimeError("data changed during formal training")
        write_new(ROOT / "benchmarks/v259_initial_validation_check.json", {"matched": True,
                  "maximum_absolute_differences": initial})
        wait_gpu_slot()
        subprocess.run([sys.executable, "-u", "scripts/audit_pure_neural_candidate.py", "--submission", OUTPUT,
                        "--label", LABEL, "--baseline-label", CONTROL], cwd=ROOT, check=True)
        verify_bound_inputs(plan)
    except Exception as error:
        write_new(output.parent / "failure.json", {"error": repr(error), "partial_outputs_preserved": True,
                                                   "implicit_resume_allowed": False})
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()

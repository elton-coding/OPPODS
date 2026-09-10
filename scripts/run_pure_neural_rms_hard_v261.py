"""V261: hard ranking under unchanged eight-expert RMS full72k protocol."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

from audit_pure_neural_candidate import fingerprint, score_paths
from probe_pure_neural_rms_hard_v261 import CPU_PROOF, DESIGN, GPU_PROOF, source_paths
from run_pure_neural_fairness_v251 import check_initial
from run_pure_neural_long_budget_v250 import command as control_command
from run_pure_neural_long_budget_v250 import data_record, require_report
from run_pure_neural_lr_v237 import ROOT, fingerprints
from run_pure_neural_rms_budget_v242 import require_audit
from run_pure_neural_rms_global_v259 import require_result as require_dependency_result
from run_storage_factorial_v254 import verify_bound_inputs, wait_gpu_slot, write_new
from train_pure_neural_rms_hard_rank_v261 import LOSS_KIND

CONTROL = "v250_eight_rms_72k"
DEPENDENCY = "v259_eight_rms_global400_18k"
DEPENDENCY_STEPS = 18000
LABEL = "v261_eight_rms_hard_rank_72k"
OUTPUT = "artifacts/pure_neural_v261/eight/hard_rank_steps72000"
PROBE = "artifacts/resource_probe/v261/eight_rms_hard_rank"
DEPENDENCY_DIR = "artifacts/pure_neural_v259/eight/global400_steps18000"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def command(*, probe=False):
    result = control_command()
    result[2] = "scripts/train_pure_neural_rms_hard_rank_v261.py"
    result[result.index("--loss-kind") + 1] = LOSS_KIND
    result[result.index("--output-dir") + 1] = PROBE if probe else OUTPUT
    if probe:
        for flag in ("--steps", "--validate-every"):
            result[result.index(flag) + 1] = "2"
    return result


def require_result(report, *, probe=False):
    if (report.get("loss_kind") != LOSS_KIND
            or report.get("train_components") != ["transmitter", "receiver"]
            or report.get("trainable_parameters") != 189551584
            or report.get("variable_payload") is not False or report.get("shared_frontend") is not False
            or any(k in report for k in ("learning_rate_schedule", "encoder_learning_rate", "microbatch_size"))):
        raise ValueError("V261 requires its RMS hard-rank loss and unchanged full-payload architecture/budget")
    if not probe:
        require_report({**report, "loss_kind": "rms_score"}, 72000)
        return
    expected = {"requested_steps": 2, "batch_size": 100, "seed": 15240, "learning_rate": 1e-5,
                "loss_kind": LOSS_KIND, "score_temperature": .5, "score_bce_weight": .05,
                "score_fairness_weight": .3, "quantile_bandwidth": .025, "validation_samples": 2000,
                "baseline_expert_map": [0, 0, 1, 1, 1, 1, 1, 1], "gpu_memory_fraction": .4}
    if (any(report.get(k) != v for k, v in expected.items())
            or [r["step"] for r in report.get("history", [])] != [0, 2]
            or report.get("gpu_peak_allocated_bytes", 0) <= 0):
        raise ValueError("incomplete or mismatched original-entry GPU probe")


def require_runtime(record, device):
    if device not in {"cpu", "cuda"}:
        raise ValueError("unsupported device")
    cuda = device == "cuda"
    expected = {"passed": True, "device": device, "weights_saved": False, "loss_kind": LOSS_KIND,
                "batch_size": 100 if cuda else 8, "parameters": 190354976,
                "trainable_parameters": 189551584, "trainable_parameter_tensors": 1296,
                "route_counts": [13]*4+[12]*4 if cuda else [1]*8,
                "mapping": [0,0,1,1,1,1,1,1], "encoder_frozen_and_unchanged": True,
                "gpu_memory_fraction": .4 if cuda else None}
    rows = record.get("steps", [])
    if (any(record.get(k) != v for k,v in expected.items()) or [r["step"] for r in rows] != [1,2]
            or any(r.get("adam_parameter_states") != 1296 or r.get("finite_gradient_tensors") != 1296
                   or r.get("loss_backward_rng_unchanged") is not True
                   or not math.isfinite(r.get("loss", float("nan")))
                   or not math.isfinite(r.get("gradient_norm_before_clip", float("nan")))
                   or r.get("gradient_norm_before_clip", 0) <= 0 for r in rows)):
        raise ValueError("incomplete all-route RMS hard-rank runtime proof")
    for row in rows:
        for key, limit in (("hard_forward_difference",2e-7), ("manual_surrogate_gradient_maximum_difference",1e-7)):
            value = row.get(key,float("inf"))
            if not math.isfinite(value) or abs(value) > limit:
                raise ValueError("hard-forward or manual surrogate-gradient contract failed")
    if cuda and record.get("gpu_peak_allocated_bytes", 0) <= 0:
        raise ValueError("missing GPU allocation evidence")
    paths = source_paths() + ([ROOT / CPU_PROOF] if cuda else [])
    if record.get("input_sha256") != fingerprints(paths):
        raise ValueError("runtime source set or hashes differ from registered implementation")


def require_design(directory, plan):
    if fingerprint(directory)["modelDesign.py"]["sha256"] != plan["input_sha256"][DESIGN]:
        raise ValueError("trained output does not use the unchanged frozen eight-expert design")


def wait_dependency(timeout):
    path = ROOT / f"benchmarks/{DEPENDENCY}_audit_offset2000.json"
    deadline = time.monotonic() + timeout
    print(json.dumps({"waiting_for_full_audit": DEPENDENCY, "steps": DEPENDENCY_STEPS}), flush=True)
    while True:
        if path.exists():
            try:
                audit = read(path)
            except json.JSONDecodeError:
                pass
            else:
                require_audit(audit, DEPENDENCY, DEPENDENCY_STEPS)
                require_dependency_result(audit["training_report"])
                if (fingerprint(ROOT / DEPENDENCY_DIR) != audit["files"]
                        or read(ROOT / DEPENDENCY_DIR / "training_report.json") != audit["training_report"]):
                    raise RuntimeError("completed V259 dependency changed")
                paths = [path, ROOT / DEPENDENCY_DIR / "training_report.json"]
                paths += [ROOT / DEPENDENCY_DIR / name for name in audit["files"]]
                paths += score_paths(ROOT / "benchmarks", DEPENDENCY, [22701, 22702, 22703], 2000)
                return {"input_sha256": fingerprints(paths), "exact_mean_final": audit["exact_mean_final"],
                        "resource_dependency_only": True, "v261_control": CONTROL}
        if time.monotonic() >= deadline:
            raise TimeoutError("V259 incomplete; inspect original process, no restart")
        time.sleep(min(30., max(.01, deadline - time.monotonic())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-timeout", type=float, default=86400)
    args = parser.parse_args()
    output, probe = ROOT / OUTPUT, ROOT / PROBE
    reserved = [output, probe, output.parent / "execution_plan.json", output.parent / "execution.lock",
                output.parent / "failure.json", output.parent / "completed_dependency.json", ROOT / GPU_PROOF,
                ROOT / "benchmarks/v261_gpu_training_probe.json", ROOT / "benchmarks/v261_initial_validation_check.json",
                ROOT / f"benchmarks/{LABEL}_audit_offset2000.json"]
    reserved += score_paths(ROOT / "benchmarks", LABEL, [22701, 22702, 22703], 2000)
    if any(p.exists() for p in reserved):
        raise FileExistsError("V261 output exists; no implicit restart/resume/overwrite")
    require_runtime(read(ROOT / CPU_PROOF), "cpu")
    control_path = ROOT / f"benchmarks/{CONTROL}_audit_offset2000.json"
    control = read(control_path)
    require_audit(control, CONTROL, 72000)
    require_report(control["training_report"], 72000)
    if fingerprint(ROOT / "artifacts/pure_neural_v250/eight/steps72000") != control["files"]:
        raise RuntimeError("frozen control changed")
    prior_path = ROOT / "artifacts/pure_neural_v260/eight/execution_plan.json"
    prior = read(prior_path)
    verify_bound_inputs(prior)
    paths = source_paths() + [control_path, ROOT / CPU_PROOF, prior_path]
    paths += [ROOT / n for n in prior["input_sha256"]]
    paths += [ROOT / "scripts/run_pure_neural_rms_hard_v261.py"]
    plan = {"label": LABEL, "control": CONTROL, "dependency": DEPENDENCY,
            "input_sha256": fingerprints(paths), "dataset": data_record(), "wait_timeout": args.wait_timeout,
            "command": command(), "probe_command": command(probe=True), "train_channel_draws": 7200000,
            "single_factor": "Only hard rank and exact hard forward score inside RMS; unchanged temperature/BCE/architecture/budget",
            "initialization": "original V227; fresh Adam, no probe/control/dependency weights reused",
            "trainable_parameters": 189551584, "automatic_promotion": False}
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
        subprocess.run([sys.executable, "-u", "scripts/probe_pure_neural_rms_hard_v261.py", "--device", "cuda"],
                       cwd=ROOT, check=True)
        require_runtime(read(ROOT / GPU_PROOF), "cuda")
        wait_gpu_slot()
        subprocess.run(plan["probe_command"], cwd=ROOT, check=True)
        require_design(probe, plan)
        report = read(probe / "training_report.json")
        require_result(report, probe=True)
        initial = check_initial(report, control["training_report"])
        write_new(ROOT / "benchmarks/v261_gpu_training_probe.json", {"purpose": "initial/runtime only, not score",
                  "report": report, "initial_validation_differences": initial})
        verify_bound_inputs(plan)
        wait_gpu_slot()
        subprocess.run(plan["command"], cwd=ROOT, check=True)
        require_design(output, plan)
        report = read(output / "training_report.json")
        require_result(report)
        initial = check_initial(report, control["training_report"])
        verify_bound_inputs(plan)
        if data_record() != plan["dataset"]:
            raise RuntimeError("data changed during formal training")
        write_new(ROOT / "benchmarks/v261_initial_validation_check.json", {"matched": True,
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

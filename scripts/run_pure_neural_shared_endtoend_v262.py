"""V262 shared-prefix x Encoder-unfreeze factorial: prove, fresh72k, paired audit."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

from analyze_shared_endtoend_factorial_v262 import analyze_groups, load_groups
from audit_pure_neural_candidate import MODEL_FILES, fingerprint, score_paths
from compare_paired_holdout import compare
from probe_pure_neural_shared_endtoend_v262 import COMPONENTS, CPU_PROOF, DESIGN, GPU_PROOF, source_paths
from recover_frozen_path_guard import frozen_path_hash
from run_frozen_confirmation_v257 import PLAN_PATH as CONFIRM_PLAN
from run_frozen_confirmation_v257 import PROTOCOL as CONFIRM_PROTOCOL
from run_frozen_confirmation_v257 import RESULT_PATH as CONFIRM_DECISION
from run_frozen_confirmation_v257 import SOURCES as CONFIRM_SOURCES
from run_frozen_confirmation_v257 import decision as confirm_decision
from run_frozen_confirmation_v257 import make_plan as confirm_plan
from run_pure_neural_endtoend_v258 import require_result as require_b
from run_pure_neural_fairness_v251 import check_initial
from run_pure_neural_long_budget_v250 import command as control_command
from run_pure_neural_long_budget_v250 import data_record, require_report
from run_pure_neural_lr_v237 import ROOT, fingerprints
from run_pure_neural_rms_budget_v242 import require_audit
from run_pure_neural_rms_hard_v261 import OUTPUT as DEPENDENCY_DIR
from run_pure_neural_rms_hard_v261 import require_result as require_dependency_result
from run_pure_neural_shared_v257 import require_result as require_a
from run_storage_factorial_v254 import verify_bound_inputs, wait_gpu_slot, write_new

LABEL = "v262_eight_rms_shared_endtoend_72k"
OUTPUT = "artifacts/pure_neural_v262/eight/shared_endtoend_steps72000"
PROBE = "artifacts/resource_probe/v262/eight_shared_endtoend"
DEPENDENCY = "v261_eight_rms_hard_rank_72k"
SOURCES = {
    "C": ("v250_eight_rms_72k", "artifacts/pure_neural_v250/eight/steps72000"),
    "A": ("v257_eight_shared_prefix8_rms_72k", "artifacts/pure_neural_v257/eight/shared8_steps72000"),
    "B": ("v258_eight_rms_endtoend_72k", "artifacts/pure_neural_v258/eight/endtoend_steps72000"),
}
FACTORIAL = ROOT / "benchmarks/v262_shared_endtoend_factorial.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def command(*, probe=False):
    result = control_command()
    result.insert(result.index("--train-components") + 1, "encoder")
    result[result.index("--model-design") + 1] = DESIGN
    result[result.index("--output-dir") + 1] = PROBE if probe else OUTPUT
    if probe:
        for flag in ("--steps", "--validate-every"):
            result[result.index(flag) + 1] = "2"
    return result


def require_result(report, *, probe=False):
    expected = {"stage": "calibrate", "train_components": COMPONENTS, "trainable_parameters": 73547840,
                "variable_payload": False, "tail_fraction": .1, "focus_prob": 0.,
                "train_snr_interval": [-20., 20.], "validation_match_train_snr": False, "shared_frontend": False}
    if (any(report.get(k) != v for k, v in expected.items())
            or any(k in report for k in ("learning_rate_schedule", "encoder_learning_rate", "microbatch_size"))):
        raise ValueError("V262 requires shared architecture/all components and original RMS sampling")
    normalized = {**report, "train_components": ["transmitter", "receiver"]}
    if probe:
        if [row["step"] for row in report.get("history", [])] != [0, 2]:
            raise ValueError("original-entry probe must complete exactly two steps")
        # Reuse the full protocol checker without allowing a partial formal report.
        normalized["history"] = [{**row, "step": 0} for row in report["history"][:1]]
        normalized["requested_steps"] = 0
        require_report(normalized, 0)
        if report.get("requested_steps") != 2:
            raise ValueError("probe budget changed")
    else:
        require_report(normalized, 72000)


def require_runtime(record, device):
    if device not in {"cpu", "cuda"}:
        raise ValueError("unsupported probe device")
    cuda = device == "cuda"
    expected = {"passed": True, "device": device, "weights_saved": False,
                "batch_size": 100 if cuda else 8, "route_counts": [13]*4+[12]*4 if cuda else [1]*8,
                "trainable_parameters": 73547840, "trainable_parameter_tensors": 532,
                "shared_parameter_alias_checks": 1024, "shared_prefix_blocks": 8,
                "gpu_memory_fraction": .4 if cuda else None}
    rows = record.get("steps", [])
    if (any(record.get(k) != v for k, v in expected.items())
            or [row.get("step") for row in rows] != [1, 2]):
        raise ValueError("incomplete shared/end-to-end runtime proof")
    for row in rows:
        expected_counts = {"encoder": (803392, 4), "transmitter": (38995360, 268), "receiver": (33749088, 260)}
        components = row.get("component_gradients", {})
        if set(components) != set(expected_counts) or row.get("adam_parameter_states") != 532:
            raise ValueError("missing component or full Adam state")
        for name, (count, tensors) in expected_counts.items():
            values = components[name]
            if (values.get("parameters") != count or values.get("gradient_tensors") != tensors
                    or not math.isfinite(values.get("gradient_l1", float("nan"))) or values["gradient_l1"] <= 0):
                raise ValueError("component gradient/count differs")
        for key in ("loss", "gradient_norm_before_clip", "encoder_maximum_parameter_change_from_initial"):
            value = row.get(key, float("nan"))
            if not math.isfinite(value) or (key != "loss" and value <= 0):
                raise ValueError("invalid optimizer update/Encoder change")
    if not cuda:
        identity = record.get("identity") or {}
        for key in ("rng_equal", "frozen_encoder_has_no_gradients", "independent_endtoend_initial_output_and_loss_equal"):
            if identity.get(key) is not True:
                raise ValueError("initial function/RNG proof missing")
        for key, limit in (("logit_maximum_difference", 0.), ("loss_difference", 0.),
                           ("transceiver_gradient_maximum_difference_before_clip", 1e-5),
                           ("independent_grouped_gradient_maximum_difference", 1e-5),
                           ("independent_encoder_gradient_maximum_difference", 1e-5)):
            value = identity.get(key, float("inf"))
            if not math.isfinite(value) or abs(value) > limit:
                raise ValueError("CPU initial/grouped gradient mismatch")
    elif record.get("gpu_peak_allocated_bytes", 0) <= 0:
        raise ValueError("missing GPU allocation proof")
    if record.get("input_sha256") != fingerprints(source_paths() + ([ROOT / CPU_PROOF] if cuda else [])):
        raise ValueError("probe source hashes changed")


def require_design(directory, plan):
    if fingerprint(directory)["modelDesign.py"]["sha256"] != frozen_path_hash(plan["input_sha256"], DESIGN):
        raise ValueError("formal design differs from frozen shared architecture")


def checked_sources():
    audits, paths = {}, []
    for arm, (label, directory) in SOURCES.items():
        path = ROOT / f"benchmarks/{label}_audit_offset2000.json"
        audit = read(path)
        require_audit(audit, label, 72000)
        if arm == "A":
            require_a(audit["training_report"])
        elif arm == "B":
            require_b(audit["training_report"])
        else:
            require_report(audit["training_report"], 72000)
        if (audit["files"] != fingerprint(ROOT / directory)
                or audit["training_report"] != read(ROOT / directory / "training_report.json")):
            raise RuntimeError("completed factorial source changed")
        paths += [path, ROOT / directory / "training_report.json"]
        paths += [ROOT / directory / name for name in MODEL_FILES]
        paths += score_paths(ROOT / "benchmarks", label, [22701, 22702, 22703], 2000)
        audits[arm] = audit
    for arm in ("A", "B"):
        comparison = audits[arm]["comparisons"][SOURCES["C"][0]]
        values = [row["delta"]["final"] for row in comparison["per_seed"]]
        values.append(comparison["paired_delta_95_percentile_interval"][0])
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("registered single-factor positive evidence is missing")
    return audits, paths


def completed_dependencies():
    path = ROOT / f"benchmarks/{DEPENDENCY}_audit_offset2000.json"
    dependency = read(path)
    require_audit(dependency, DEPENDENCY, 72000)
    require_dependency_result(dependency["training_report"])
    if (dependency["files"] != fingerprint(ROOT / DEPENDENCY_DIR)
            or dependency["training_report"] != read(ROOT / DEPENDENCY_DIR / "training_report.json")):
        raise RuntimeError("V261 must complete the actual full training/audit, including any explicit recovery")
    registered = read(CONFIRM_PLAN)
    if registered != confirm_plan():
        raise RuntimeError("V257 confirmation freeze changed")
    confirmation_paths = [ROOT / f"benchmarks/{row[1]}_audit_offset8000.json" for row in CONFIRM_SOURCES]
    confirmation_audits = [read(path) for path in confirmation_paths]
    for audit, source in zip(confirmation_audits, registered["sources"], strict=True):
        if (audit.get("label") != source["label"] or audit.get("protocol") != CONFIRM_PROTOCOL
                or audit.get("files") != source["files"]
                or audit.get("training_report") != read(ROOT / source["directory"] / "training_report.json")
                or [row.get("seed") for row in audit.get("per_seed", [])] != [52701, 52702, 52703]
                or any(row.get("short_outputs") != 0 or row.get("min_output_length") != 1152
                       or row.get("max_output_length") != 1152 or row.get("samples") != 2000
                       or row.get("scores") != 4000 for row in audit.get("per_seed", []))):
            raise RuntimeError("confirmation source/protocol/full-payload audit differs")
    candidate_confirmation = confirmation_audits[1]
    confirmation_caches = [score_paths(ROOT / "benchmarks", row[1], [52701, 52702, 52703], 8000)
                           for row in CONFIRM_SOURCES]
    if compare(*confirmation_caches) != candidate_confirmation["comparisons"][CONFIRM_SOURCES[0][1]]:
        raise RuntimeError("V257 confirmation paired caches do not reproduce")
    decision = read(CONFIRM_DECISION)
    if confirm_decision(candidate_confirmation) != decision:
        raise RuntimeError("V257 frozen confirmation decision differs from its complete audit")
    paths = [path, ROOT / DEPENDENCY_DIR / "training_report.json", CONFIRM_PLAN, CONFIRM_DECISION, *confirmation_paths]
    paths += [ROOT / DEPENDENCY_DIR / name for name in MODEL_FILES]
    paths += score_paths(ROOT / "benchmarks", DEPENDENCY, [22701, 22702, 22703], 2000)
    for row in CONFIRM_SOURCES:
        paths += score_paths(ROOT / "benchmarks", row[1], [52701, 52702, 52703], 8000)
    return {"input_sha256": fingerprints(paths), "resource_dependency_only": True,
            "v261_audit_mean": dependency["exact_mean_final"],
            "v257_confirmation_mean": candidate_confirmation["exact_mean_final"],
            "v257_confirmation_supports_promotion": decision["supports_current_cohort_promotion"]}


def wait_dependencies(timeout):
    deadline = time.monotonic() + timeout
    path = ROOT / f"benchmarks/{DEPENDENCY}_audit_offset2000.json"
    print({"waiting_for_full_v261_audit_and_v257_confirmation": True}, flush=True)
    while True:
        if path.exists() and CONFIRM_DECISION.exists():
            try:
                return completed_dependencies()
            except json.JSONDecodeError:
                pass
        if time.monotonic() >= deadline:
            raise TimeoutError("dependencies incomplete; inspect original jobs without restarting them")
        time.sleep(min(30., max(.01, deadline-time.monotonic())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-timeout", type=float, default=86400)
    args = parser.parse_args()
    output, probe = ROOT / OUTPUT, ROOT / PROBE
    base = output.parent
    audit_path = ROOT / f"benchmarks/{LABEL}_audit_offset2000.json"
    reserved = [output, probe, base / "execution_plan.json", base / "execution.lock", base / "failure.json",
                base / "completed_dependency.json", base / "deferred.json", ROOT / GPU_PROOF, FACTORIAL, audit_path,
                ROOT / "benchmarks/v262_gpu_training_probe.json", ROOT / "benchmarks/v262_initial_validation_check.json"]
    reserved += score_paths(ROOT / "benchmarks", LABEL, [22701, 22702, 22703], 2000)
    if any(path.exists() for path in reserved):
        raise FileExistsError("V262 output exists; no implicit restart, resume, or overwrite")
    require_runtime(read(ROOT / CPU_PROOF), "cpu")
    audits, paths = checked_sources()
    paths += source_paths() + [ROOT / CPU_PROOF, CONFIRM_PLAN]
    bound_paths = [ROOT / "artifacts/pure_neural_v261/eight/execution_plan.json",
                   ROOT / "artifacts/pure_neural_v260/eight/path_guard_recovery/execution_plan.json", CONFIRM_PLAN]
    for path in bound_paths:
        bound = read(path)
        verify_bound_inputs(bound)
        paths += [path, *[ROOT / name for name in bound["input_sha256"]]]
    paths += [ROOT / f"scripts/{name}" for name in (
        "run_pure_neural_shared_endtoend_v262.py", "analyze_shared_endtoend_factorial_v262.py",
        "recover_frozen_path_guard.py", "run_frozen_confirmation_v257.py", "run_pure_neural_shared_v257.py",
        "run_pure_neural_endtoend_v258.py", "run_pure_neural_rms_hard_v261.py", "audit_pure_neural_candidate.py",
        "evaluate_submission.py", "compare_paired_holdout.py", "run_storage_factorial_v254.py")]
    plan = {"label": LABEL, "sources": SOURCES, "input_sha256": fingerprints(paths), "dataset": data_record(),
            "command": command(), "probe_command": command(probe=True), "train_channel_draws": 7200000,
            "wait_timeout": args.wait_timeout, "trainable_parameters": 73547840, "trainable_parameter_tensors": 532,
            "factorial": "C V250; A V257 prefix sharing; B V258 Encoder unfreeze; AB V262 both",
            "criteria": "AB-C, AB-A and AB-B each all three totals >0 and channel-bootstrap95 lower bound >0",
            "factorial_bootstrap_seed": 262, "factorial_bootstrap_repeats": 2000,
            "initialization": "original V227 fresh Adam; no source/probe/dependency checkpoint reuse",
            "automatic_promotion": False}
    base.mkdir(parents=True, exist_ok=True)
    lock = base / "execution.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        write_new(base / "execution_plan.json", plan)
        print({"plan": str(base / "execution_plan.json"), "frozen_inputs": len(plan["input_sha256"])}, flush=True)
        dependency = wait_dependencies(args.wait_timeout)
        write_new(base / "completed_dependency.json", dependency)
        if max(*[a["exact_mean_final"] for a in audits.values()], dependency["v261_audit_mean"], dependency["v257_confirmation_mean"]) >= 69:
            write_new(base / "deferred.json", {"reason": "source reached local69; require confirmation/compliance review"})
            return
        verify_bound_inputs(plan)
        if data_record() != plan["dataset"]:
            raise RuntimeError("dataset changed while waiting for dependencies")
        wait_gpu_slot(timeout=86400)
        subprocess.run([sys.executable, "-u", "scripts/probe_pure_neural_shared_endtoend_v262.py", "--device", "cuda"], cwd=ROOT, check=True)
        require_runtime(read(ROOT / GPU_PROOF), "cuda")
        wait_gpu_slot(timeout=86400)
        subprocess.run(plan["probe_command"], cwd=ROOT, check=True)
        require_design(probe, plan)
        report = read(probe / "training_report.json")
        require_result(report, probe=True)
        initial = check_initial(report, audits["C"]["training_report"])
        write_new(ROOT / "benchmarks/v262_gpu_training_probe.json", {"purpose": "initial/runtime only, not score",
                  "report": report, "initial_validation_differences": initial})
        verify_bound_inputs(plan)
        wait_gpu_slot(timeout=86400)
        subprocess.run(plan["command"], cwd=ROOT, check=True)
        require_design(output, plan)
        report = read(output / "training_report.json")
        require_result(report)
        initial = check_initial(report, audits["C"]["training_report"])
        verify_bound_inputs(plan)
        if data_record() != plan["dataset"]:
            raise RuntimeError("dataset changed during formal training")
        write_new(ROOT / "benchmarks/v262_initial_validation_check.json", {"matched": True, "maximum_absolute_differences": initial})
        wait_gpu_slot(timeout=86400)
        subprocess.run([sys.executable, "-u", "scripts/audit_pure_neural_candidate.py", "--submission", OUTPUT,
                        "--label", LABEL, "--baseline-label", SOURCES["C"][0], "--control-label", SOURCES["A"][0]], cwd=ROOT, check=True)
        audit = read(audit_path)
        require_audit(audit, LABEL, 72000)
        if audit["files"] != fingerprint(output) or audit["training_report"] != report:
            raise RuntimeError("formal audit source changed")
        labels = {arm: row[0] for arm, row in SOURCES.items()} | {"AB": LABEL}
        cache_paths = {arm: score_paths(ROOT / "benchmarks", label, [22701, 22702, 22703], 2000) for arm, label in labels.items()}
        result = analyze_groups(load_groups(cache_paths))
        raw_means = {arm: a["exact_mean_final"] for arm, a in audits.items()} | {"AB": audit["exact_mean_final"]}
        if any(abs(result["arm_mean_final_from_cache"][arm] - mean) > 1e-4 for arm, mean in raw_means.items()):
            raise RuntimeError("raw evaluator and paired-cache scores disagree")
        result.update({"labels": labels, "raw_evaluator_mean_final": raw_means,
                       "input_sha256": fingerprints([path for group in cache_paths.values() for path in group])})
        verify_bound_inputs(plan)
        verify_bound_inputs(dependency)
        write_new(FACTORIAL, result)
        print({"factorial": str(FACTORIAL), "mean": audit["exact_mean_final"], "eligible": result["eligible_for_new_confirmation"]}, flush=True)
    except Exception as error:
        write_new(base / "failure.json", {"error": repr(error), "partial_outputs_preserved": True, "implicit_resume_allowed": False})
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()

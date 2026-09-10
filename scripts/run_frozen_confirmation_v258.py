"""Preregister V258 confirmation before V257's shared reference window is evaluated."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import run_frozen_confirmation_v257 as reference
from audit_pure_neural_candidate import MODEL_FILES, fingerprint, score_paths
from compare_paired_holdout import compare, metrics
from run_pure_neural_endtoend_v258 import LABEL as SOURCE_LABEL
from run_pure_neural_endtoend_v258 import OUTPUT as SOURCE_DIR
from run_pure_neural_endtoend_v258 import require_result
from run_pure_neural_lr_v237 import ROOT, fingerprints
from run_pure_neural_rms_budget_v242 import require_audit
from run_storage_factorial_v254 import verify_bound_inputs, wait_gpu_slot, write_new

from oppods.data import deterministic_split_indices

LABEL = "v258_confirm"
PROTOCOL = reference.PROTOCOL
SEEDS = reference.SEEDS
OFFSET = reference.OFFSET
PLAN_PATH = ROOT / "benchmarks/v258_confirmation_plan.json"
RESULT_PATH = ROOT / "benchmarks/v258_confirmation_decision.json"
FAILURE_PATH = ROOT / "benchmarks/v258_confirmation_failure.json"
BINDING_PATH = ROOT / "benchmarks/v258_confirmation_reference.json"
AUDIT_PATH = ROOT / f"benchmarks/{LABEL}_audit_offset{OFFSET}.json"
REFERENCES = tuple(row[1] for row in reference.SOURCES)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def reference_outputs():
    return [reference.RESULT_PATH, ROOT / "benchmarks/v257_confirmation_failure.json",
            *[ROOT / f"benchmarks/{label}_audit_offset{OFFSET}.json" for label in REFERENCES],
            *[path for label in REFERENCES for path in score_paths(ROOT / "benchmarks", label, SEEDS, OFFSET)]]


def require_unseen_reference():
    if any(path.exists() for path in reference_outputs()):
        raise FileExistsError("reference confirmation already produced outputs; cannot preregister retrospectively")


def command():
    return [sys.executable, "-u", "scripts/audit_pure_neural_candidate.py", "--submission", SOURCE_DIR,
            "--label", LABEL, "--baseline-label", REFERENCES[0], "--control-label", REFERENCES[1],
            "--test-offset", str(OFFSET), "--samples", "2000", "--noise-seeds", *map(str, SEEDS)]


def comparison_gate(value):
    rows = value["per_seed"]
    interval = value["paired_delta_95_percentile_interval"]
    if [row["noise_seed"] for row in rows] != SEEDS or len(interval) != 2:
        raise ValueError("need three ordered seeds and a two-sided interval")
    values = [value["mean_delta"], *interval, *[row["delta"][key] for row in rows
                                              for key in ("final", "efficiency", "p10")]]
    if not all(math.isfinite(v) for v in values) or interval[0] > interval[1]:
        raise ValueError("nonfinite or reversed confirmation statistics")
    return {"passes": min(row["delta"]["final"] for row in rows) > 0 and interval[0] > 0,
            "mean_delta": value["mean_delta"], "paired_delta_95_interval": interval,
            "per_seed_metric_deltas": [row["delta"] for row in rows]}


def decision(audit):
    if (audit.get("label") != LABEL or audit.get("protocol") != PROTOCOL
            or set(audit.get("comparisons", {})) != set(REFERENCES)
            or not math.isfinite(audit["exact_mean_final"])):
        raise ValueError("wrong candidate, reference arms or confirmation protocol")
    comparisons = {label: comparison_gate(audit["comparisons"][label]) for label in REFERENCES}
    return {"candidate_exact_mean_final": audit["exact_mean_final"], "comparisons": comparisons,
            "supports_improvement_over_frozen_v250": comparisons[REFERENCES[0]]["passes"],
            "supports_preference_over_both_frozen_references": all(c["passes"] for c in comparisons.values()),
            "automatic_promotion": False, "online_confirmation": False,
            "whole_project_blindness_certified": False, "ancestor_training_provenance_certified": False,
            "caveat": "shared-window conditional cohort confirmation; 1812/2000 historical evaluation overlap; "
                      "not independent of V257 confirmation, not adjusted for historical model selection"}


def checked_caches(labels, *, offset=OFFSET, seeds=SEEDS):
    expected_ids = deterministic_split_indices(100000, seed=1176)["test"][offset:offset + 2000].repeat(2)
    reference_snr, exact = {}, {}
    for label in labels:
        records = []
        for seed, path in zip(seeds, score_paths(ROOT / "benchmarks", label, seeds, offset), strict=True):
            with np.load(path, allow_pickle=False) as cache:
                for key, expected in (("split_seed", 1176), ("test_offset", offset), ("noise_seed", seed)):
                    if cache[key].size != 1 or cache[key].dtype.kind not in "iu" or int(cache[key].item()) != expected:
                        raise ValueError("wrong cache metadata")
                values = {key: cache[key].copy() for key in ("data_index", "length", "score", "snr")}
            if (any(v.shape != (4000,) for v in values.values()) or values["data_index"].dtype.kind not in "iu"
                    or not np.array_equal(values["data_index"], expected_ids)
                    or not np.all(values["length"] == 1152)
                    or not np.isfinite(values["score"]).all() or np.any(values["score"] < 0)
                    or np.any(values["score"] > 100) or not np.isfinite(values["snr"]).all()
                    or np.any(values["snr"] < -20) or np.any(values["snr"] > 20)):
                raise ValueError("wrong actual channel window or incomplete/invalid full payload")
            if seed in reference_snr and not np.array_equal(reference_snr[seed], values["snr"]):
                raise ValueError("unpaired reference/candidate SNR")
            reference_snr[seed] = values["snr"]
            records.append(metrics(values["score"].astype(np.float64)))
        exact[label] = records
    return exact


def checked_audit(audit, source, cache_metrics):
    rows = audit.get("per_seed", [])
    if (audit.get("label") != source["label"] or audit.get("protocol") != PROTOCOL
            or audit.get("files") != source["files"]
            or audit.get("training_report") != read(ROOT / source["directory"] / "training_report.json")
            or [row.get("seed") for row in rows] != SEEDS
            or any(row.get("short_outputs") != 0 or row.get("min_output_length") != 1152
                   or row.get("max_output_length") != 1152 or row.get("samples") != 2000
                   or row.get("scores") != 4000 for row in rows)):
        raise ValueError("confirmation audit source/protocol/full payload differs")
    for row, cached in zip(rows, cache_metrics, strict=True):
        for key, cache_key in (("final", "final"), ("efficiency", "efficiency"), ("fairness", "p10")):
            if not math.isfinite(row[key]) or abs(row[key] - cached[cache_key]) > 1e-4:
                raise ValueError("raw metrics do not reproduce from full paired cache")
    mean = sum(row["final"] for row in rows) / 3
    if not math.isfinite(audit["exact_mean_final"]) or abs(mean - audit["exact_mean_final"]) > 1e-10:
        raise ValueError("raw audit mean differs from its three seeds")


def make_plan():
    prior = read(reference.PLAN_PATH)
    if reference.make_plan() != prior:
        raise RuntimeError("V257 frozen reference plan changed")
    path = ROOT / f"benchmarks/{SOURCE_LABEL}_audit_offset2000.json"
    audit = read(path)
    require_audit(audit, SOURCE_LABEL, 72000)
    require_result(audit["training_report"])
    files = fingerprint(ROOT / SOURCE_DIR)
    if files != audit["files"] or read(ROOT / SOURCE_DIR / "training_report.json") != audit["training_report"]:
        raise RuntimeError("V258 completed model/report changed")
    candidate_caches = score_paths(ROOT / "benchmarks", SOURCE_LABEL, [22701, 22702, 22703], 2000)
    baseline_caches = score_paths(ROOT / "benchmarks", reference.SOURCES[0][0], [22701, 22702, 22703], 2000)
    checked_caches([row[0] for row in reference.SOURCES] + [SOURCE_LABEL], offset=2000, seeds=[22701, 22702, 22703])
    selected = compare(baseline_caches, candidate_caches)
    if selected != audit["comparisons"][reference.SOURCES[0][0]]:
        raise RuntimeError("V258 original paired selection statistics do not reproduce")
    if (min(row["delta"]["final"] for row in selected["per_seed"]) <= 0
            or selected["paired_delta_95_percentile_interval"][0] <= 0):
        raise ValueError("V258 failed original selection gate versus V250")
    paths = [reference.PLAN_PATH, *[ROOT / p for p in prior["input_sha256"]], path, *candidate_caches,
             ROOT / SOURCE_DIR / "training_report.json", *[ROOT / SOURCE_DIR / n for n in MODEL_FILES],
             ROOT / "scripts/run_frozen_confirmation_v258.py"]
    for name in ("execution_plan.json", "completed_dependency.json"):
        bound_path = ROOT / "artifacts/pure_neural_v258/eight" / name
        bound = read(bound_path)
        verify_bound_inputs(bound)
        paths += [bound_path, *[ROOT / p for p in bound["input_sha256"]]]
    return {"candidate": {"label": LABEL, "directory": SOURCE_DIR, "files": files},
            "input_sha256": fingerprints(paths), "dataset": prior["dataset"], "protocol": PROTOCOL,
            "reference_sources": prior["sources"], "command": command(),
            "preregister_before_any_reference_confirmation_outputs": True,
            "selection_gate_versus_v250_reproduced": True,
            "criteria": "each comparison needs all three paired total deltas >0 and 95% lower bound >0; "
                        "preference over both V250 and V257 requires both comparisons to pass",
            "v257_selection_comparison": "inconclusive; not treated as proof V258 already beats V257",
            "bootstrap": {"unit": "channel, both UEs together; shared across noise replicas", "repeats": 2000, "seed": 227},
            "historical_inventory_window": prior["historical_inventory_window"],
            "resource_policy": "wait for complete V257 confirmation, reuse frozen reference caches, then acquire GPU slot",
            "independent_of_v257_confirmation": False, "automatic_promotion": False, "online_confirmation": False}


def checked_reference(plan):
    if make_plan() != plan:
        raise RuntimeError("registered candidate/reference freeze changed")
    cached = checked_caches(REFERENCES)
    audits = []
    paths = [reference.RESULT_PATH]
    for source in plan["reference_sources"]:
        path = ROOT / f"benchmarks/{source['label']}_audit_offset{OFFSET}.json"
        audit = read(path)
        checked_audit(audit, source, cached[source["label"]])
        audits.append(audit)
        paths += [path, *score_paths(ROOT / "benchmarks", source["label"], SEEDS, OFFSET)]
    reproduced = compare(*[score_paths(ROOT / "benchmarks", label, SEEDS, OFFSET) for label in REFERENCES])
    if (reproduced != audits[1]["comparisons"][REFERENCES[0]]
            or reference.decision(audits[1]) != read(reference.RESULT_PATH)):
        raise RuntimeError("reference paired statistics or frozen decision do not reproduce")
    return {"input_sha256": fingerprints(paths), "reference_labels": list(REFERENCES),
            "reuse_without_reselecting_models_or_window": True}


def wait_reference(plan, timeout):
    deadline = time.monotonic() + timeout
    while True:
        if reference.RESULT_PATH.exists():
            try:
                return checked_reference(plan)
            except json.JSONDecodeError:
                pass
        if (ROOT / "benchmarks/v257_confirmation_failure.json").exists():
            raise RuntimeError("reference confirmation failed; preserve outputs and inspect its original process")
        if time.monotonic() >= deadline:
            raise TimeoutError("reference confirmation incomplete; do not restart healthy upstream jobs")
        time.sleep(min(30., max(.01, deadline - time.monotonic())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--wait-timeout", type=float, default=86400)
    args = parser.parse_args()
    reserved = [RESULT_PATH, FAILURE_PATH, BINDING_PATH, AUDIT_PATH,
                *score_paths(ROOT / "benchmarks", LABEL, SEEDS, OFFSET)]
    if any(path.exists() for path in reserved):
        raise FileExistsError("confirmation output exists; no repeat, overwrite or implicit resume")
    if args.register_only:
        require_unseen_reference()
    plan = make_plan()
    if args.register_only:
        require_unseen_reference()
        write_new(PLAN_PATH, plan)
        print(json.dumps({"registered": str(PLAN_PATH), "evaluation_started": False}), flush=True)
        return
    if read(PLAN_PATH) != plan:
        raise RuntimeError("freeze differs from preregistered plan")
    lock = ROOT / "artifacts/v258_confirmation.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        print(json.dumps({"waiting_for_full_v257_confirmation": True, "gpu_evaluation_started": False}), flush=True)
        bound = wait_reference(plan, args.wait_timeout)
        write_new(BINDING_PATH, bound)
        wait_gpu_slot(timeout=args.wait_timeout)
        if make_plan() != plan:
            raise RuntimeError("source changed before confirmation")
        verify_bound_inputs(bound)
        subprocess.run(plan["command"], cwd=ROOT, check=True)
        if make_plan() != plan:
            raise RuntimeError("source changed during confirmation")
        verify_bound_inputs(bound)
        audit = read(AUDIT_PATH)
        cached = checked_caches((*REFERENCES, LABEL))
        checked_audit(audit, plan["candidate"], cached[LABEL])
        for label in REFERENCES:
            reproduced = compare(score_paths(ROOT / "benchmarks", label, SEEDS, OFFSET),
                                 score_paths(ROOT / "benchmarks", LABEL, SEEDS, OFFSET))
            if reproduced != audit["comparisons"][label]:
                raise RuntimeError("candidate paired statistics do not reproduce")
        result = decision(audit)
        write_new(RESULT_PATH, result)
        print(json.dumps(result), flush=True)
    except Exception as error:
        write_new(FAILURE_PATH, {"error": repr(error), "partial_outputs_preserved": True, "implicit_resume_allowed": False})
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()

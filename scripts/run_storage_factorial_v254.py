"""Complete the preregistered V254 storage factorial; no training or promotion here."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile

import numpy as np
import psutil
from audit_pure_neural_candidate import MODEL_FILES, ROOT, fingerprint, score_paths
from build_submission import ARCHIVE_MEMBERS, package_submission
from compare_factorial_holdout import factorial
from compare_paired_holdout import metrics
from convert_storage_v254 import ARMS, require_source_audit, transformed_training_report
from run_pure_neural_lr_v237 import fingerprints
from run_pure_neural_pair_v253 import check_training_slot
from run_pure_neural_rms_budget_v242 import PROTOCOL, require_audit

BASE = ROOT / "artifacts/pure_neural_v254/storage"
FACTORIAL = ROOT / "benchmarks/v254_expert_storage_factorial.json"
LABELS = {"C": ARMS["eight"]["source_label"], "A": ARMS["sixteen"]["source_label"],
          "B": ARMS["eight"]["label"], "AB": ARMS["sixteen"]["label"]}
DIRECTORIES = {"C": ROOT / ARMS["eight"]["source"], "A": ROOT / ARMS["sixteen"]["source"],
               "B": ROOT / ARMS["eight"]["output"], "AB": ROOT / ARMS["sixteen"]["output"]}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def audit_path(group):
    return ROOT / f"benchmarks/{LABELS[group]}_audit_offset2000.json"


def scores(group):
    return score_paths(ROOT / "benchmarks", LABELS[group], PROTOCOL["noise_seeds"], 2000)


def verify_bound_inputs(record):
    expected = record["input_sha256"]
    if fingerprints([ROOT / p for p in expected]) != expected:
        raise RuntimeError("frozen inputs changed; do not continue this experiment")


def verify_source(arm):
    group = "C" if arm == "eight" else "A"
    audit = read(audit_path(group))
    require_source_audit(audit, arm)
    if (fingerprint(DIRECTORIES[group]) != audit["files"]
            or read(DIRECTORIES[group] / "training_report.json") != audit["training_report"]):
        raise RuntimeError("source model/report changed since full audit")
    return audit


def require_conversion_record(record, arm, source, report):
    config = ARMS[arm]
    check = record.get("cpu_check", {})
    if (record.get("label") != config["label"] or record.get("source_label") != config["source_label"]
            or record.get("source_files") != source["files"] or record.get("new_training_steps") != 0
            or record.get("inference_dtype") != "float32" or record.get("passed_runtime_check") is not True
            or record.get("encoder_copied_exactly") is not True
            or check.get("loaded_parameters_dtype") != "float32"
            or check.get("all_stored_values_loaded_exactly") is not True
            or check.get("routes") != list(range(config["num_experts"]))):
        raise ValueError("conversion proof is missing, mismatched or not float32 inference")
    provenance = record["provenance"]
    if (provenance["source_directory"] != config["source"]
            or provenance["output_directory"] != config["output"]
            or report != transformed_training_report(source["training_report"], provenance)):
        raise ValueError("converted report must explicitly preserve source training history only")
    if any(record["files"][name] != source["files"][name] for name in ("modelDesign.py", "encoder.pth")):
        raise ValueError("design/Encoder must remain byte identical")


def verify_conversion(arm, source):
    config = ARMS[arm]
    directory = ROOT / config["output"]
    record = read(ROOT / f"benchmarks/{config['label']}_conversion.json")
    require_conversion_record(record, arm, source, read(directory / "training_report.json"))
    verify_bound_inputs(record)
    if fingerprint(directory) != record["files"]:
        raise RuntimeError("converted files changed")
    return record


def require_scored_payload(audit, group):
    require_audit(audit, LABELS[group], 72000)
    expected = {"split_seed": 1176, "test_offset": 2000, "samples": 2000, "scores": 4000,
                "short_outputs": 0, "min_output_length": 1152, "max_output_length": 1152}
    for seed, row in zip(PROTOCOL["noise_seeds"], audit["per_seed"], strict=True):
        if row.get("seed") != seed or any(row.get(key) != value for key, value in expected.items()):
            raise ValueError("incomplete/wrong fixed-seed payload audit")
        if not all(np.isfinite(row[key]) for key in ("efficiency", "fairness", "final")):
            raise ValueError("nonfinite audit score")
    if (not np.isfinite(audit["exact_mean_final"])
            or abs(audit["exact_mean_final"] - np.mean([r["final"] for r in audit["per_seed"]])) > 1e-10):
        raise ValueError("audit mean inconsistent with exact evaluator output")


def checked_scores(group, audit):
    require_scored_payload(audit, group)
    for seed, path, row in zip(PROTOCOL["noise_seeds"], scores(group), audit["per_seed"], strict=True):
        with np.load(path) as data:
            if (any(data[key].shape != (4000,) for key in ("score", "snr", "length", "data_index"))
                    or not np.all(data["length"] == 1152) or not np.isfinite(data["score"]).all()
                    or not np.isfinite(data["snr"]).all()
                    or any(int(data[key]) != value for key, value in (
                        ("noise_seed", seed), ("split_seed", 1176), ("test_offset", 2000)))):
                raise ValueError("incomplete or wrong cached scores")
            ids = data["data_index"].reshape(2000, 2)
            if not np.array_equal(ids[:, 0], ids[:, 1]) or len(np.unique(ids[:, 0])) != 2000:
                raise ValueError("cache must contain 2000 distinct paired channels")
            computed = metrics(data["score"].astype(np.float64))
            if any(abs(computed[key] - row[other]) > 1e-4 for key, other in (
                    ("efficiency", "efficiency"), ("p10", "fairness"), ("final", "final"))):
                raise ValueError("float32 cached scores do not reproduce exact evaluator metrics")
    return scores(group)


def verify_archive(path, files):
    with zipfile.ZipFile(path) as archive:
        if tuple(archive.namelist()) != ARCHIVE_MEMBERS or archive.testzip() is not None:
            raise ValueError("ZIP member order or CRC failed")
        for member, name in zip(ARCHIVE_MEMBERS, MODEL_FILES, strict=True):
            with archive.open(member) as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != files[name]["sha256"] or archive.getinfo(member).file_size != files[name]["bytes"]:
                raise ValueError("ZIP member differs from evaluated file")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    size = path.stat().st_size
    return {"path": str(path), "bytes": size, "sha256": digest, "members_crc_hash_verified": True,
            "within_conservative_1GB_limit": size <= 1_000_000_000,
            "organizer_hardware_runtime_certified": False}


def half_blocks(groups):
    """Descriptive prespecified channel halves; call after factorial pairing verification."""
    result = []
    for i, seed in enumerate(PROTOCOL["noise_seeds"]):
        arrays = {}
        for group, paths in groups.items():
            with np.load(paths[i]) as data:
                arrays[group] = data["score"].astype(np.float64).reshape(2000, 2)
        for start, stop in ((0, 1000), (1000, 2000)):
            values = {g: metrics(a[start:stop]) for g, a in arrays.items()}
            result.append({"noise_seed": seed, "channel_slice": [start, stop], "metrics": values,
                           "deltas_vs_C": {g: {k: values[g][k] - values["C"][k] for k in values["C"]}
                                           for g in ("A", "B", "AB")}})
    return result


def confirmation_gate(contrast, package):
    return (len(contrast["per_seed"]) == 3
            and all(row["delta"]["final"] > 0 for row in contrast["per_seed"])
            and contrast["paired_delta_95_percentile_interval"][0] > 0
            and package["members_crc_hash_verified"] is True
            and package["within_conservative_1GB_limit"] is True)


def wait_completed_source(timeout):
    deadline = time.monotonic() + timeout
    print(json.dumps({"waiting_for_full_audit": LABELS["A"], "steps": 72000}), flush=True)
    while True:
        if audit_path("A").exists():
            try:
                read(audit_path("A"))
            except json.JSONDecodeError:
                pass  # Writer may still be flushing its final JSON; never use a checkpoint instead.
            else:
                return verify_source("sixteen")
        if time.monotonic() >= deadline:
            raise TimeoutError("A full audit unavailable; inspect existing training, do not duplicate/restart")
        time.sleep(min(30., max(.01, deadline - time.monotonic())))


def wait_gpu_slot(timeout=3600):
    deadline = time.monotonic() + timeout
    while True:
        busy = False
        try:
            check_training_slot()
        except RuntimeError:
            busy = True
        # A completed audit can coexist briefly with another runner's GPU evaluation.
        for process in psutil.process_iter(["name", "cmdline"]):
            try:
                cmd = process.info["cmdline"] or []
                if (str(process.info["name"]).lower() == "python.exe"
                        and any(str(p).replace("\\", "/").split("/")[-1] == "evaluate_submission.py" for p in cmd)):
                    busy = True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if not busy:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("GPU audit slot unavailable; preserve artifacts and inspect existing jobs")
        time.sleep(min(30., max(.01, deadline - time.monotonic())))


def run_audit(group):
    wait_gpu_slot()
    command = [sys.executable, "-u", "scripts/audit_pure_neural_candidate.py", "--submission",
               str(DIRECTORIES[group]), "--label", LABELS[group], "--baseline-label", LABELS["C"]]
    if group == "AB":
        command += ["--control-label", LABELS["A"]]
    subprocess.run(command, cwd=ROOT, check=True)
    return read(audit_path(group))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-timeout", type=float, default=86400)
    args = parser.parse_args()
    reserved = [BASE / "execution_plan.json", BASE / "execution.lock", BASE / "completed_source_A.json",
                BASE / "decision.json", BASE / "failure.json", BASE / "packages", FACTORIAL, DIRECTORIES["AB"],
                ROOT / f"benchmarks/{LABELS['AB']}_conversion.json"]
    reserved += [p for g in ("B", "AB") for p in [audit_path(g), *scores(g)]]
    if any(p.exists() for p in reserved):
        raise FileExistsError("storage pipeline outputs exist; inspect, no implicit resume/overwrite")
    control = verify_source("eight")
    existing_b = verify_conversion("eight", control)
    checked_scores("C", control)
    a_plan_path = ROOT / "artifacts/pure_neural_v254/sixteen/execution_plan.json"
    a_plan = read(a_plan_path)
    verify_bound_inputs(a_plan)
    paths = [ROOT / p for p in a_plan["input_sha256"]]
    paths += [ROOT / p for p in existing_b["input_sha256"]]
    paths += [a_plan_path, audit_path("C"), *scores("C")]
    paths += [ROOT / f"benchmarks/{LABELS['B']}_conversion.json"]
    paths += [DIRECTORIES[g] / name for g in ("C", "B") for name in (*MODEL_FILES, "training_report.json")]
    paths += [DIRECTORIES["B"] / "conversion_plan.json"]
    paths += [ROOT / "scripts" / p for p in ("run_storage_factorial_v254.py", "convert_storage_v254.py",
              "compare_factorial_holdout.py", "compare_paired_holdout.py", "build_submission.py")]
    plan = {"labels": LABELS, "input_sha256": fingerprints(paths), "wait_timeout": args.wait_timeout,
            "protocol": PROTOCOL, "factor_A": "8 to 16 nested joint SNR experts, same72k training",
            "factor_B": "TxRx half storage only; float32 inference; no additional training",
            "blocks": [[0, 1000], [1000, 2000]], "bootstrap_repeats": 2000,
            "registered_control": "V250", "automatic_promotion": False,
            "gate": "AB-C positive all3seeds and paired95lower>0, verified ZIP<=1e9; then new confirmation"}
    with (BASE / "execution.lock").open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        write_new(BASE / "execution_plan.json", plan)
        print(json.dumps({"plan": str(BASE / "execution_plan.json"), "frozen_inputs": len(plan["input_sha256"])}), flush=True)
        source_a = wait_completed_source(args.wait_timeout)
        verify_bound_inputs(plan)
        checked_scores("A", source_a)
        a_paths = [audit_path("A"), *scores("A")]
        a_paths += [DIRECTORIES["A"] / n for n in (*MODEL_FILES, "training_report.json")]
        source_record = {"input_sha256": fingerprints(a_paths), "files": source_a["files"]}
        write_new(BASE / "completed_source_A.json", source_record)
        subprocess.run([sys.executable, "-u", "scripts/convert_storage_v254.py", "--arm", "sixteen"], cwd=ROOT, check=True)
        converted = {"B": existing_b, "AB": verify_conversion("sixteen", source_a)}
        audits = {"C": control, "A": source_a}
        for group in ("B", "AB"):
            verify_bound_inputs(plan)
            verify_bound_inputs(source_record)
            verify_conversion("eight" if group == "B" else "sixteen", control if group == "B" else source_a)
            audits[group] = run_audit(group)
            if (audits[group]["files"] != converted[group]["files"]
                    or audits[group]["training_report"] != read(DIRECTORIES[group] / "training_report.json")):
                raise RuntimeError("converted audit provenance changed")
        groups = {g: checked_scores(g, audits[g]) for g in LABELS}
        result = factorial(groups, repeats=2000, factor_a=plan["factor_A"], factor_b=plan["factor_B"])
        result["exact_evaluator_mean_final"] = {g: a["exact_mean_final"] for g, a in audits.items()}
        result["prespecified_half_blocks_descriptive"] = half_blocks(groups)
        result["score_input_sha256"] = fingerprints([p for group in groups.values() for p in group])
        packages = {}
        for group in ("A", "B", "AB"):
            path = BASE / "packages" / f"{group}_{LABELS[group]}.zip"
            if path.exists():
                raise FileExistsError("package output appeared; refusing overwrite")
            package_submission(DIRECTORIES[group], path)
            packages[group] = verify_archive(path, audits[group]["files"])
        result["packages"] = packages
        verify_bound_inputs(plan)
        verify_bound_inputs(source_record)
        if any(fingerprint(DIRECTORIES[g]) != audits[g]["files"] for g in LABELS):
            raise RuntimeError("model changed during factorial/package checks")
        if fingerprints([p for group in groups.values() for p in group]) != result["score_input_sha256"]:
            raise RuntimeError("score inputs changed during factorial")
        write_new(FACTORIAL, result)
        eligible = confirmation_gate(result["contrasts_vs_control"]["AB"], packages["AB"])
        decision = {"eligible_for_frozen_confirmation_vs_registered_v250": eligible,
                    "factorial_report": str(FACTORIAL), "exact_mean_final": result["exact_evaluator_mean_final"],
                    "automatic_promotion": False, "online_confirmation": False,
                    "whole_project_blindness_certified": False, "ancestor_training_provenance_certified": False,
                    "next": "Compare with then-current champion if changed; preregister unused current-cohort confirmation window before evaluation.",
                    "caveat": "A score never substitutes for AB; source FP32 training history is not converted validation."}
        write_new(BASE / "decision.json", decision)
        print(json.dumps(decision), flush=True)
    except Exception as error:
        failure = BASE / "failure.json"
        if not failure.exists():
            write_new(failure, {"error": repr(error), "partial_outputs_preserved": True, "implicit_resume_allowed": False})
        raise
    finally:
        (BASE / "execution.lock").unlink()


if __name__ == "__main__":
    main()

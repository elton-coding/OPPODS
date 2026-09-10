"""Explicit recovery of a completed probe's Windows path-key guard failure.

Never modifies the frozen original runner, restarts live work, resumes trained
weights, or reruns a completed probe. Formal training must not have started.
"""
from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import PureWindowsPath

import psutil
from audit_pure_neural_candidate import MODEL_FILES, ROOT, fingerprint, score_paths
from run_pure_neural_lr_v237 import fingerprints
from run_storage_factorial_v254 import verify_bound_inputs, wait_gpu_slot, write_new

MODULES = {"v260": "run_pure_neural_receiver_v260", "v261": "run_pure_neural_rms_hard_v261"}


def frozen_path_hash(mapping, relative_path):
    target = PureWindowsPath(relative_path)
    if target.is_absolute() or ".." in target.parts:
        raise ValueError("expected a workspace-relative path without traversal")
    matches = [(key, value) for key, value in mapping.items() if PureWindowsPath(key) == target]
    if len(matches) != 1:
        raise ValueError("frozen path must have exactly one unambiguous spelling")
    return matches[0][1]


def require_design(directory, original, design):
    if fingerprint(directory)["modelDesign.py"]["sha256"] != frozen_path_hash(original["input_sha256"], design):
        raise ValueError("output design differs from the original frozen bytes")


def require_failure(failure, design):
    if failure != {"error": repr(KeyError(design)), "partial_outputs_preserved": True,
                   "implicit_resume_allowed": False}:
        raise ValueError("recovery only permits the known path-key failure after the original probe")


def require_original_stopped(module):
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = process.info["cmdline"] or []
            if str(process.info["name"]).lower() != "python.exe":
                continue
            leaves = [str(p).replace("\\", "/").split("/")[-1] for p in cmd]
            if module.__name__ + ".py" in leaves:
                raise RuntimeError(f"original runner still alive: {process.info['pid']}")
            if "--output-dir" in cmd and str(cmd[cmd.index("--output-dir")+1]).replace("\\", "/") in {
                    module.OUTPUT, module.PROBE}:
                raise RuntimeError(f"original trainer/probe still alive: {process.info['pid']}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def make_plan(module, version):
    require_original_stopped(module)
    base = (ROOT / module.OUTPUT).parent
    if (ROOT / module.OUTPUT).exists() or (base / "execution.lock").exists():
        raise FileExistsError("formal output or original lock exists; no implicit resume")
    original = module.read(base / "execution_plan.json")
    failure = module.read(base / "failure.json")
    require_failure(failure, module.DESIGN)
    verify_bound_inputs(original)
    dependency = module.read(base / "completed_dependency.json")
    verify_bound_inputs(dependency)
    if original["command"] != module.command() or original["probe_command"] != module.command(probe=True):
        raise ValueError("original commands changed")
    if module.data_record() != original["dataset"]:
        raise RuntimeError("dataset or split changed")
    module.require_runtime(module.read(ROOT / module.GPU_PROOF), "cuda")
    report = module.read(ROOT / module.PROBE / "training_report.json")
    module.require_result(report, probe=True)
    require_design(ROOT / module.PROBE, original, module.DESIGN)
    control = module.read(ROOT / f"benchmarks/{module.CONTROL}_audit_offset2000.json")
    module.require_audit(control, module.CONTROL, 72000)
    module.require_report(control["training_report"], 72000)
    initial = module.check_initial(report, control["training_report"])
    paths = [base / name for name in ("execution_plan.json", "completed_dependency.json", "failure.json")]
    paths += [ROOT / name for bound in (original, dependency) for name in bound["input_sha256"]]
    paths += [ROOT / module.PROBE / name for name in (*MODEL_FILES, "training_report.json")]
    paths += [ROOT / module.GPU_PROOF, ROOT / "scripts/recover_frozen_path_guard.py"]
    return {"version": version, "input_sha256": fingerprints(paths), "dataset": original["dataset"],
            "command": original["command"], "original_failure": failure,
            "initial_validation_differences": initial, "original_probe_report": report,
            "scope": "only path-key lookup corrected; reuse completed proof, not its weights",
            "fresh_initialization": "unchanged original V227 command; fresh Adam, no resume",
            "automatic_promotion": False, "original_failure_preserved": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=MODULES, required=True)
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    module = importlib.import_module(MODULES[args.version])
    base = (ROOT / module.OUTPUT).parent / "path_guard_recovery"
    plan_path = base / "execution_plan.json"
    probe_evidence = ROOT / f"benchmarks/{args.version}_gpu_training_probe.json"
    initial_path = ROOT / f"benchmarks/{args.version}_initial_validation_check.json"
    audit_path = ROOT / f"benchmarks/{module.LABEL}_audit_offset2000.json"
    reserved = [probe_evidence, initial_path, audit_path, base / "failure.json", base / "completed.json"]
    reserved += score_paths(ROOT / "benchmarks", module.LABEL, [22701, 22702, 22703], 2000)
    if any(path.exists() for path in reserved):
        raise FileExistsError("recovery output already exists; preserve and inspect")
    plan = make_plan(module, args.version)
    if args.register_only:
        base.mkdir(parents=True, exist_ok=True)
        write_new(plan_path, plan)
        print({"registered_recovery": str(plan_path), "frozen_inputs": len(plan["input_sha256"]),
               "formal_training_started": False}, flush=True)
        return
    if module.read(plan_path) != plan:
        raise RuntimeError("recovery differs from registered freeze")
    lock = base / "execution.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        wait_gpu_slot(timeout=86400)
        if make_plan(module, args.version) != plan:
            raise RuntimeError("recovery inputs changed while waiting")
        write_new(probe_evidence, {"purpose": "completed original probe, recovered path lookup; not score",
                                  "report": plan["original_probe_report"],
                                  "initial_validation_differences": plan["initial_validation_differences"]})
        subprocess.run(plan["command"], cwd=ROOT, check=True)
        verify_bound_inputs(plan)
        original = module.read((ROOT / module.OUTPUT).parent / "execution_plan.json")
        require_design(ROOT / module.OUTPUT, original, module.DESIGN)
        report = module.read(ROOT / module.OUTPUT / "training_report.json")
        module.require_result(report)
        control = module.read(ROOT / f"benchmarks/{module.CONTROL}_audit_offset2000.json")
        initial = module.check_initial(report, control["training_report"])
        if module.data_record() != plan["dataset"]:
            raise RuntimeError("dataset changed during recovered formal run")
        write_new(initial_path, {"matched": True, "maximum_absolute_differences": initial})
        wait_gpu_slot(timeout=86400)
        subprocess.run([sys.executable, "-u", "scripts/audit_pure_neural_candidate.py", "--submission", module.OUTPUT,
                        "--label", module.LABEL, "--baseline-label", module.CONTROL], cwd=ROOT, check=True)
        verify_bound_inputs(plan)
        audit = module.read(audit_path)
        module.require_audit(audit, module.LABEL, 72000)
        if audit["training_report"] != report or audit["files"] != fingerprint(ROOT / module.OUTPUT):
            raise RuntimeError("completed recovery audit differs from formal source")
        write_new(base / "completed.json", {"audit": str(audit_path.relative_to(ROOT)),
                  "exact_mean_final": audit["exact_mean_final"], "original_failure_preserved": True,
                  "formal_training_steps": 72000, "automatic_promotion": False})
    except Exception as error:
        write_new(base / "failure.json", {"error": repr(error), "partial_outputs_preserved": True})
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()

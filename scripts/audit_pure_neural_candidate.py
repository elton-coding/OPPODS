"""Audit one completed training run, binding exact metrics to immutable weights.

This does not promote a model or interpret a local score as a leaderboard score.
Run only after the candidate training process has completed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from compare_paired_holdout import compare

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = ("modelDesign.py", "encoder.pth", "transmitter.pth", "receiver.pth")


def fingerprint(directory: Path) -> dict[str, dict[str, int | str]]:
    result = {}
    for name in MODEL_FILES:
        path = directory / name
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        result[name] = {"sha256": digest, "bytes": path.stat().st_size}
    return result


def score_paths(directory: Path, label: str, seeds: list[int], offset: int) -> list[Path]:
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", label):
        raise ValueError("label must contain only letters, digits, underscores, or hyphens")
    return [directory / f"{label}_holdout1176_offset{offset}_noise{seed}.npz" for seed in seeds]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--baseline-label", default="v227")
    parser.add_argument("--control-label")
    parser.add_argument("--noise-seeds", type=int, nargs="+", default=[22701, 22702, 22703])
    parser.add_argument("--test-offset", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--benchmarks", type=Path, default=ROOT / "benchmarks")
    args = parser.parse_args()
    submission = args.submission.resolve()
    training_path = submission / "training_report.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    if not training.get("history") or training["history"][-1]["step"] <= 0:
        raise ValueError("candidate needs a completed, nonempty training report")
    if len(set(args.noise_seeds)) != len(args.noise_seeds):
        raise ValueError("noise seeds must be distinct")
    benchmarks = args.benchmarks.resolve()
    paths = score_paths(benchmarks, args.label, args.noise_seeds, args.test_offset)
    report_path = benchmarks / f"{args.label}_audit_offset{args.test_offset}.json"
    if report_path.exists() or any(path.exists() for path in paths):
        raise FileExistsError("audit labels are immutable; choose a new label instead of overwriting")
    baselines = {
        name: score_paths(benchmarks, name, args.noise_seeds, args.test_offset)
        for name in (args.baseline_label, args.control_label) if name
    }
    for group in baselines.values():
        for path in group:
            if not path.is_file():
                raise FileNotFoundError(f"paired baseline is missing: {path}")
    initial = fingerprint(submission)
    results = []
    benchmarks.mkdir(parents=True, exist_ok=True)
    for seed, path in zip(args.noise_seeds, paths, strict=True):
        command = [sys.executable, str(ROOT / "scripts/evaluate_submission.py"),
                   "--submission", str(submission), "--samples", str(args.samples),
                   "--split-seed", "1176", "--test-offset", str(args.test_offset),
                   "--seed", str(seed), "--progress-every", "0", "--scores-out", str(path)]
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
        result = json.loads(completed.stdout)
        if fingerprint(submission) != initial:
            raise RuntimeError("candidate changed during evaluation; discard this audit label")
        if result["samples"] != args.samples or result["scores"] != 2 * args.samples:
            raise RuntimeError("incomplete evaluation")
        results.append({"command": command, **result})
        print(json.dumps({"label": args.label, **result}), flush=True)
    report = {
        "label": args.label,
        "submission": str(submission),
        "files": initial,
        "training_report": training,
        "protocol": {"split_seed": 1176, "test_offset": args.test_offset,
                     "samples": args.samples, "noise_seeds": args.noise_seeds},
        "per_seed": results,
        "exact_mean_final": sum(item["final"] for item in results) / len(results),
        "comparisons": {name: compare(group, paths) for name, group in baselines.items()},
        "caveat": "Local repeated-selection audit, not an unseen confirmation or an online score.",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "exact_mean_final": report["exact_mean_final"],
                      "deltas": {name: value["mean_delta"] for name, value in report["comparisons"].items()}},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

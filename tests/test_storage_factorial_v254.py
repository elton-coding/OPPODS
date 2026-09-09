import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_storage_factorial_v254 as pipeline


def source_audit():
    return json.loads(pipeline.audit_path("C").read_text(encoding="utf-8"))


def test_existing_b_proof_preserves_source_training_not_new_validation():
    record = json.loads((ROOT / "benchmarks/v254_eight_fp16_storage_conversion.json").read_text(encoding="utf-8"))
    audit = source_audit()
    report = pipeline.transformed_training_report(audit["training_report"], record["provenance"])
    pipeline.require_conversion_record(record, "eight", audit, report)
    for key, value in (("inference_dtype", "float16"), ("new_training_steps", 72000), ("source_label", "wrong")):
        changed = copy.deepcopy(record)
        changed[key] = value
        with pytest.raises(ValueError):
            pipeline.require_conversion_record(changed, "eight", audit, report)
    with pytest.raises(ValueError):
        pipeline.require_conversion_record(record, "eight", audit, audit["training_report"])


@pytest.mark.parametrize("mutation", ["partial", "wrong_seed", "short", "nan", "wrong_mean"])
def test_full_scored_payload_guards(mutation):
    audit = source_audit()
    pipeline.require_scored_payload(audit, "C")
    if mutation == "partial":
        audit["training_report"]["history"].pop()
    elif mutation == "wrong_seed":
        audit["per_seed"][0]["seed"] = 99
    elif mutation == "short":
        audit["per_seed"][0]["min_output_length"] = 576
    elif mutation == "nan":
        audit["exact_mean_final"] = float("nan")
    else:
        audit["exact_mean_final"] += .1
    with pytest.raises(ValueError):
        pipeline.require_scored_payload(audit, "C")


def test_write_new_is_exclusive_and_preserves_existing(tmp_path):
    path = tmp_path / "plan.json"
    pipeline.write_new(path, {"first": 1})
    with pytest.raises(FileExistsError):
        pipeline.write_new(path, {"second": 2})
    assert pipeline.read(path) == {"first": 1}


def test_archive_verification_binds_actual_members_not_just_size(tmp_path):
    directory = tmp_path / "source"
    directory.mkdir()
    files = {}
    for name in pipeline.MODEL_FILES:
        payload = name.encode()
        (directory / name).write_bytes(payload)
        files[name] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    path = tmp_path / "test.zip"
    pipeline.package_submission(directory, path)
    evidence = pipeline.verify_archive(path, files)
    assert evidence["members_crc_hash_verified"] and evidence["within_conservative_1GB_limit"]
    files["receiver.pth"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        pipeline.verify_archive(path, files)


def test_gate_requires_all_seeds_positive_interval_and_deployable_package():
    contrast = {"per_seed": [{"delta": {"final": .1}} for _ in range(3)],
                "paired_delta_95_percentile_interval": [.01, .2]}
    package = {"members_crc_hash_verified": True, "within_conservative_1GB_limit": True}
    assert pipeline.confirmation_gate(contrast, package)
    package["within_conservative_1GB_limit"] = False
    assert not pipeline.confirmation_gate(contrast, package)
    package["within_conservative_1GB_limit"] = True
    contrast["per_seed"][1]["delta"]["final"] = 0
    assert not pipeline.confirmation_gate(contrast, package)
    contrast["per_seed"][1]["delta"]["final"] = .1
    contrast["paired_delta_95_percentile_interval"][0] = 0
    assert not pipeline.confirmation_gate(contrast, package)


def test_prespecified_blocks_do_not_mix_channel_halves(tmp_path):
    groups = {}
    for group, delta in (("C", 0), ("A", 1), ("B", 2), ("AB", 4)):
        groups[group] = []
        for seed in pipeline.PROTOCOL["noise_seeds"]:
            path = tmp_path / f"{group}_{seed}.npz"
            np.savez(path, score=np.concatenate([np.full(2000, 60. + delta), np.full(2000, 80. + delta)]))
            groups[group].append(path)
    blocks = pipeline.half_blocks(groups)
    assert len(blocks) == 6
    assert blocks[0]["metrics"]["C"]["final"] == 60.
    assert blocks[1]["metrics"]["C"]["final"] == 80.
    assert all(abs(row["deltas_vs_C"]["AB"]["final"] - 4.) < 1e-10 for row in blocks)


def test_completed_control_cache_is_reusable_without_gpu():
    assert len(pipeline.checked_scores("C", source_audit())) == 3


def test_registered_output_labels_distinct_and_not_main():
    assert len(set(pipeline.LABELS.values())) == 4
    assert len(set(pipeline.DIRECTORIES.values())) == 4
    assert all(p.name != "modelSubmit" for p in pipeline.DIRECTORIES.values())


def test_gpu_slot_rejects_third_training_and_existing_evaluator(monkeypatch):
    monkeypatch.setattr(pipeline.psutil, "process_iter", lambda _: [])
    monkeypatch.setattr(pipeline, "check_training_slot", lambda: [123])
    pipeline.wait_gpu_slot(timeout=0)

    def two_jobs():
        raise RuntimeError("two jobs")

    monkeypatch.setattr(pipeline, "check_training_slot", two_jobs)
    with pytest.raises(TimeoutError):
        pipeline.wait_gpu_slot(timeout=0)
    monkeypatch.setattr(pipeline, "check_training_slot", list)
    process = type("Process", (), {"info": {"name": "python.exe", "cmdline": ["python", "evaluate_submission.py"]}})()
    monkeypatch.setattr(pipeline.psutil, "process_iter", lambda _: [process])
    with pytest.raises(TimeoutError):
        pipeline.wait_gpu_slot(timeout=0)


def test_waiter_does_not_accept_file_existence_or_partial_json(monkeypatch, tmp_path):
    path = tmp_path / "audit.json"
    monkeypatch.setattr(pipeline, "audit_path", lambda _: path)
    with pytest.raises(TimeoutError):
        pipeline.wait_completed_source(0)
    path.write_text("{", encoding="utf-8")
    with pytest.raises(TimeoutError):
        pipeline.wait_completed_source(0)


def test_waiter_requires_full_source_validator(monkeypatch, tmp_path):
    path = tmp_path / "audit.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(pipeline, "audit_path", lambda _: path)

    def incomplete(_):
        raise ValueError("incomplete full-budget training")

    monkeypatch.setattr(pipeline, "verify_source", incomplete)
    with pytest.raises(ValueError, match="incomplete"):
        pipeline.wait_completed_source(0)

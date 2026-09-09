from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import probe_pure_neural_shared_memory_v257 as memory
import run_pure_neural_shared_v257 as runner


def original_report():
    return json.loads((ROOT / "benchmarks/v250_eight_rms_72k_audit_offset2000.json")
                      .read_text(encoding="utf-8"))["training_report"]


def test_shared_runner_only_changes_architecture_and_output():
    expected = runner.control_command()
    expected[expected.index("--model-design")+1] = runner.DESIGN
    expected[expected.index("--output-dir")+1] = runner.OUTPUT
    assert runner.command() == expected
    assert runner.command()[2] == "scripts/train_pure_neural_rms_v239.py"
    probe = runner.command(probe=True)
    assert probe[probe.index("--steps")+1] == "2"
    assert probe[probe.index("--batch-size")+1] == "100"
    assert probe[probe.index("--validation-samples")+1] == "2000"


def test_full_guard_requires_shared_architecture_and_complete_constant72k():
    source = original_report()
    report = copy.deepcopy(source)
    with pytest.raises(ValueError):
        runner.require_result(report)
    report["trainable_parameters"] = 72744448
    runner.require_result(report)  # Synthetic schema fixture, never a new score.
    assert source["trainable_parameters"] == 189551584
    for key, value in (("score_temperature", .25), ("score_fairness_weight", .5),
                       ("learning_rate_schedule", {}), ("focus_prob", .5), ("variable_payload", True),
                       ("trainable_parameters", 189551584)):
        broken = {**report, key: value}
        with pytest.raises(ValueError):
            runner.require_result(broken)
    report["history"].pop()
    with pytest.raises(ValueError):
        runner.require_result(report)


def test_initial_probe_schema_cannot_replace_formal_training():
    report = original_report()
    report["trainable_parameters"] = 72744448
    report["requested_steps"] = 2
    report["history"] = [report["history"][0], {**report["history"][0], "step": 2}]
    runner.require_result(report, probe=True)
    with pytest.raises(ValueError):
        runner.require_result(report)
    report["validation_samples"] = 2
    with pytest.raises(ValueError):
        runner.require_result(report, probe=True)


def test_forced_gpu_cases_cover_eight_routes_without_cuda():
    snr = memory.forced_snr(torch.device("cpu"))
    module = memory.load_model_design(ROOT / runner.DESIGN)
    assert tuple(snr.shape) == (100, 2)
    assert torch.bincount(module._expert_indices(snr.amin(1))).tolist() == [13]*4+[12]*4
    assert memory.MEMORY_FRACTION == .4


def test_memory_guard_requires_aliases_and_complete_unique_adam_state(monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "verify_bound_inputs", lambda r: calls.append(r))
    record = {"passed": True, "batch_size": 100, "trainable_parameters": 72744448,
              "route_counts": [13]*4+[12]*4, "weights_saved": False, "gpu_memory_fraction": .4,
              "shared_parameter_alias_checks_after_cuda": 1024, "gpu_peak_allocated_bytes": 1,
              "steps": [{"step": i, "adam_parameter_states": 528, "gradient_tensors": 528} for i in (1, 2)]}
    runner.require_memory(record)
    assert len(calls) == 1
    for key, value in (("shared_parameter_alias_checks_after_cuda", 0), ("route_counts", [100]+[0]*7),
                       ("steps", record["steps"][:1]), ("weights_saved", True)):
        with pytest.raises(ValueError):
            runner.require_memory({**record, key: value})
    broken = copy.deepcopy(record)
    broken["steps"][1]["adam_parameter_states"] = 1296
    with pytest.raises(ValueError):
        runner.require_memory(broken)


def test_wait_does_not_accept_missing_or_partial_storage_decision(monkeypatch, tmp_path):
    path = tmp_path / "decision.json"
    monkeypatch.setattr(runner, "DEPENDENCY_DECISION", path)
    with pytest.raises(TimeoutError):
        runner.wait_dependency(0)
    path.write_text("{", encoding="utf-8")
    with pytest.raises(TimeoutError):
        runner.wait_dependency(0)
    called = []
    monkeypatch.setattr(runner, "completed_storage", lambda value: called.append(value) or {"checked": True})
    path.write_text('{"test_fixture": true}', encoding="utf-8")
    assert runner.wait_dependency(0) == {"checked": True}
    assert called == [{"test_fixture": True}]


def test_wait_reports_dependency_failure_without_starting_training(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "DEPENDENCY_DECISION", tmp_path / "decision.json")
    (tmp_path / "failure.json").write_text('{"error":"test"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="storage reported failure"):
        runner.wait_dependency(0)

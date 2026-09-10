import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_pure_neural_rms_hard_v261 as runner


def control_report():
    report = json.loads((ROOT / "benchmarks/v250_eight_rms_72k_audit_offset2000.json").read_text())["training_report"]
    report["loss_kind"] = runner.LOSS_KIND
    return report


def cpu_proof(monkeypatch):
    record = json.loads((ROOT / runner.CPU_PROOF).read_text())
    monkeypatch.setattr(runner, "fingerprints", lambda paths: record["input_sha256"])
    return record


def test_only_entry_loss_label_and_output_change_in_full72k_command():
    expected = runner.control_command()
    expected[2] = "scripts/train_pure_neural_rms_hard_rank_v261.py"
    expected[expected.index("--loss-kind")+1] = runner.LOSS_KIND
    expected[expected.index("--output-dir")+1] = runner.OUTPUT
    assert runner.command() == expected
    assert expected[expected.index("--model-design")+1] == runner.DESIGN
    probe = runner.command(probe=True)
    assert probe[probe.index("--steps")+1] == probe[probe.index("--validate-every")+1] == "2"
    assert probe[probe.index("--validation-samples")+1] == "2000"


def test_reports_require_all_validation_points_and_full_gpu_probe():
    report = control_report()  # Schema fixture only; not V261 performance.
    before = copy.deepcopy(report)
    runner.require_result(report)
    assert report == before
    report["requested_steps"] = 2
    report["history"] = [report["history"][0], {**report["history"][1], "step": 2}]
    runner.require_result(report, probe=True)
    report["history"].pop()
    with pytest.raises(ValueError):
        runner.require_result(report, probe=True)


@pytest.mark.parametrize("key,value", [
    ("loss_kind", "rms_score"), ("loss_kind", "hard_rank_score"), ("requested_steps", 36000), ("score_temperature", .25), ("score_bce_weight", .01),
    ("trainable_parameters", 190354976), ("train_components", ["encoder", "transmitter", "receiver"]),
    ("batch_size", 400), ("learning_rate_schedule", {}), ("encoder_learning_rate", 1e-5),
    ("microbatch_size", 100), ("shared_frontend", True), ("variable_payload", True),
])
def test_report_rejects_other_factors(key, value):
    report = control_report()
    report[key] = value
    with pytest.raises(ValueError):
        runner.require_result(report)


def test_real_cpu_proof_and_synthetic_gpu_schema(monkeypatch):
    record = cpu_proof(monkeypatch)
    runner.require_runtime(record, "cpu")
    gpu = copy.deepcopy(record)  # Synthetic schema, never used as runtime evidence.
    gpu.update(device="cuda", batch_size=100, route_counts=[13]*4+[12]*4,
               gpu_memory_fraction=.4, gpu_peak_allocated_bytes=1)
    runner.require_runtime(gpu, "cuda")
    gpu["steps"][0]["adam_parameter_states"] -= 1
    with pytest.raises(ValueError):
        runner.require_runtime(gpu, "cuda")


@pytest.mark.parametrize("failure", ["hash","grad","forward","rng","route","encoder","finite","count","kind"])
def test_runtime_proof_rejects_missing_or_wrong_invariants(monkeypatch, failure):
    record = cpu_proof(monkeypatch)
    if failure == "hash":
        monkeypatch.setattr(runner,"fingerprints",lambda paths: {})
    elif failure == "grad":
        record["steps"][0]["manual_surrogate_gradient_maximum_difference"] = .001
    elif failure == "forward":
        record["steps"][0]["hard_forward_difference"] = .001
    elif failure == "rng":
        record["steps"][0]["loss_backward_rng_unchanged"] = False
    elif failure == "route":
        record["route_counts"][0] = 0
    elif failure == "encoder":
        record["encoder_frozen_and_unchanged"] = False
    elif failure == "finite":
        record["steps"][0]["loss"] = float("nan")
    elif failure == "count":
        record["steps"][0]["finite_gradient_tensors"] = 1295
    elif failure == "kind":
        record["loss_kind"] = "hard_rank_score"
    with pytest.raises(ValueError):
        runner.require_runtime(record,"cpu")


def test_trained_design_must_match_frozen_source(monkeypatch):
    monkeypatch.setattr(runner, "fingerprint", lambda directory: {"modelDesign.py": {"sha256": "actual"}})
    runner.require_design(ROOT, {"input_sha256": {runner.DESIGN: "actual"}})
    with pytest.raises(ValueError):
        runner.require_design(ROOT, {"input_sha256": {runner.DESIGN: "other"}})


@pytest.mark.parametrize("contents", [None, "{"])
def test_missing_or_partial_predecessor_does_not_release_queue(tmp_path, monkeypatch, contents):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    if contents is not None:
        path = tmp_path / f"benchmarks/{runner.DEPENDENCY}_audit_offset2000.json"
        path.parent.mkdir()
        path.write_text(contents)
    with pytest.raises(TimeoutError, match="V259 incomplete"):
        runner.wait_dependency(0)


def test_dependency_is_matched_global18k_but_candidate_keeps72k():
    assert runner.DEPENDENCY == "v259_eight_rms_global400_18k"
    assert runner.DEPENDENCY_STEPS == 18000
    command = runner.command()
    assert command[command.index("--steps")+1] == "72000"
    assert command[command.index("--batch-size")+1] == "100"


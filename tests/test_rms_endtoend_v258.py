import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_pure_neural_endtoend_v258 as runner


def fixture_report():
    source = json.loads((ROOT / "benchmarks/v250_eight_rms_72k_audit_offset2000.json").read_text())["training_report"]
    report = copy.deepcopy(source)
    report["train_components"] = ["encoder", "transmitter", "receiver"]
    report["trainable_parameters"] = 190354976
    return report


def test_only_unfreezing_and_output_differ_from_full72k_control():
    expected = runner.control_command()
    expected.insert(expected.index("--train-components") + 1, "encoder")
    expected[expected.index("--output-dir") + 1] = runner.OUTPUT
    assert runner.command() == expected
    assert expected[2] == "scripts/train_pure_neural_rms_v239.py"
    probe = runner.command(probe=True)
    for flag, value in (("--steps", "2"), ("--validate-every", "2"), ("--batch-size", "100"),
                        ("--validation-samples", "2000"), ("--score-temperature", "0.5")):
        assert probe[probe.index(flag) + 1] == value


def test_full_and_probe_schemas_accept_only_complete_fixtures_without_mutation():
    report = fixture_report()  # A schema fixture, not a V258 performance claim.
    before = copy.deepcopy(report)
    runner.require_result(report)
    assert report == before
    probe = copy.deepcopy(report)
    probe["requested_steps"] = 2
    probe["history"] = [probe["history"][0], {**probe["history"][1], "step": 2}]
    runner.require_result(probe, probe=True)
    probe["history"].pop()
    with pytest.raises(ValueError):
        runner.require_result(probe, probe=True)


@pytest.mark.parametrize("field,value", [
    ("train_components", ["transmitter", "receiver"]), ("trainable_parameters", 189551584),
    ("score_temperature", .25), ("loss_kind", "soft_score"), ("seed", 15241),
    ("encoder_learning_rate", 5e-5), ("learning_rate_schedule", {}), ("variable_payload", True),
])
def test_report_rejects_other_ablation_factors(field, value):
    report = fixture_report()
    report[field] = value
    with pytest.raises(ValueError):
        runner.require_result(report)


def test_report_rejects_incomplete_or_missing_middle_validation():
    report = fixture_report()
    report["history"].pop(35)
    with pytest.raises(ValueError):
        runner.require_result(report)


def runtime_fixture(monkeypatch):
    # The numeric evidence is real; these tests mutate copies to exercise schema gates.
    record = json.loads((ROOT / runner.CPU_PROOF).read_text())
    monkeypatch.setattr(runner, "verify_bound_inputs", lambda _: None)
    return record


def test_runtime_proof_requires_actual_encoder_gradients_and_updates(monkeypatch):
    record = runtime_fixture(monkeypatch)
    runner.require_runtime(record, "cpu")
    for changed in ("gradient", "update", "adam", "nan", "identity"):
        bad = copy.deepcopy(record)
        if changed == "gradient":
            bad["steps"][0]["component_gradients"]["encoder"]["gradient_l1"] = 0.
        elif changed == "update":
            bad["steps"][0]["encoder_maximum_parameter_change_from_initial"] = 0.
        elif changed == "adam":
            bad["steps"][1]["adam_parameter_states"] -= 1
        elif changed == "nan":
            bad["steps"][0]["component_gradients"]["encoder"]["gradient_l1"] = float("nan")
        else:
            bad["identity"]["rng_equal"] = False
        with pytest.raises(ValueError):
            runner.require_runtime(bad, "cpu")


def test_cuda_gate_requires_all_routes_full_batch_and_resource_evidence(monkeypatch):
    record = runtime_fixture(monkeypatch)
    with pytest.raises(ValueError):
        runner.require_runtime(record, "cuda")
    record.update(device="cuda", batch_size=100, route_counts=[13] * 4 + [12] * 4,
                  gpu_memory_fraction=.4, gpu_peak_allocated_bytes=1, identity=None)
    runner.require_runtime(record, "cuda")  # Synthetic schema fixture, not actual CUDA proof.
    record["route_counts"][-1] = 0
    with pytest.raises(ValueError):
        runner.require_runtime(record, "cuda")


def test_wait_does_not_accept_missing_partial_or_wrong_dependency(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    with pytest.raises(TimeoutError):
        runner.wait_dependency(0)
    path = tmp_path / f"benchmarks/{runner.DEPENDENCY}_audit_offset2000.json"
    path.parent.mkdir()
    path.write_text("{", encoding="utf-8")
    with pytest.raises(TimeoutError):
        runner.wait_dependency(0)
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        runner.wait_dependency(0)

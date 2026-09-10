import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_pure_neural_shared_endtoend_v262 as runner


def report(probe=False):
    return {"stage": "calibrate", "train_components": ["encoder", "transmitter", "receiver"],
            "trainable_parameters": 73547840, "variable_payload": False, "tail_fraction": .1,
            "focus_prob": 0., "train_snr_interval": [-20., 20.], "validation_match_train_snr": False,
            "shared_frontend": False, "requested_steps": 2 if probe else 72000,
            "batch_size": 100, "seed": 15240, "learning_rate": 1e-5, "loss_kind": "rms_score",
            "score_temperature": .5, "score_bce_weight": .05, "score_fairness_weight": .3,
            "quantile_bandwidth": .025, "validation_samples": 2000,
            "baseline_expert_map": [0, 0, 1, 1, 1, 1, 1, 1], "gpu_memory_fraction": .4,
            "gpu_peak_allocated_bytes": 1,
            "history": [{"step": n} for n in ([0, 2] if probe else range(0, 72001, 1000))]}


def test_only_shared_design_and_encoder_training_change():
    cmd = runner.command()
    assert cmd[cmd.index("--model-design")+1] == "research/pure_neural_v257/modelDesign.py"
    start = cmd.index("--train-components")+1
    assert cmd[start:start+3] == ["encoder", "transmitter", "receiver"]
    for flag, value in (("--steps", "72000"), ("--learning-rate", "1e-5"), ("--batch-size", "100"),
                        ("--loss-kind", "rms_score"), ("--score-temperature", "0.5")):
        assert cmd[cmd.index(flag)+1] == value
    assert cmd[cmd.index("--baseline-dir")+1] == "artifacts/pure_neural_v227/joint_low"
    probe = runner.command(probe=True)
    assert probe[probe.index("--steps")+1] == "2"
    assert probe[probe.index("--output-dir")+1] != cmd[cmd.index("--output-dir")+1]


def test_complete_formal_and_two_step_probe_are_separate():
    runner.require_result(report())
    runner.require_result(report(True), probe=True)
    with pytest.raises(ValueError):
        runner.require_result(report(True))
    with pytest.raises(ValueError):
        runner.require_result(report(), probe=True)


@pytest.mark.parametrize("key,value", [
    ("train_components", ["transmitter", "receiver"]), ("trainable_parameters", 190354976),
    ("requested_steps", 36000), ("batch_size", 400), ("seed", 1), ("learning_rate", 5e-5),
    ("loss_kind", "rms_hard_rank_score"), ("score_temperature", .25), ("score_bce_weight", 0),
    ("score_fairness_weight", .5), ("variable_payload", True), ("shared_frontend", True),
    ("microbatch_size", 100), ("learning_rate_schedule", "cosine"), ("encoder_learning_rate", 5e-5),
    ("gpu_peak_allocated_bytes", 0), ("validation_samples", 100), ("focus_prob", .5),
])
def test_other_factors_and_incomplete_reports_rejected(key, value):
    value_report = report()
    value_report[key] = value
    with pytest.raises(ValueError):
        runner.require_result(value_report)


def test_missing_validation_checkpoint_rejected():
    value = report()
    value["history"].pop(5)
    with pytest.raises(ValueError):
        runner.require_result(value)


@pytest.fixture
def cpu(monkeypatch):
    value = copy.deepcopy(runner.read(runner.ROOT / runner.CPU_PROOF))
    value["input_sha256"] = {"synthetic": "test-only"}
    monkeypatch.setattr(runner, "fingerprints", lambda paths: {"synthetic": "test-only"})
    return value


def test_cpu_contract_and_synthetic_gpu_schema(cpu):
    runner.require_runtime(cpu, "cpu")
    gpu = copy.deepcopy(cpu)
    gpu.update(device="cuda", batch_size=100, route_counts=[13]*4+[12]*4,
               gpu_memory_fraction=.4, gpu_peak_allocated_bytes=100, identity=None)
    runner.require_runtime(gpu, "cuda")


@pytest.mark.parametrize("case", ["state_count", "alias", "encoder_update", "nan_loss", "component_count",
                                 "missing_component", "group_gradient", "source", "saved_weights"])
def test_incomplete_runtime_proofs_rejected(cpu, case):
    if case == "state_count":
        cpu["steps"][0]["adam_parameter_states"] = 531
    elif case == "alias":
        cpu["shared_parameter_alias_checks"] = 0
    elif case == "encoder_update":
        cpu["steps"][0]["encoder_maximum_parameter_change_from_initial"] = 0
    elif case == "nan_loss":
        cpu["steps"][0]["loss"] = float("nan")
    elif case == "component_count":
        cpu["steps"][0]["component_gradients"]["receiver"]["gradient_tensors"] = 259
    elif case == "missing_component":
        cpu["steps"][0]["component_gradients"].pop("encoder")
    elif case == "group_gradient":
        cpu["identity"]["independent_grouped_gradient_maximum_difference"] = .1
    elif case == "source":
        cpu["input_sha256"] = {}
    else:
        cpu["weights_saved"] = True
    with pytest.raises(ValueError):
        runner.require_runtime(cpu, "cpu")


def test_design_lookup_with_actual_platform_fingerprint_key(monkeypatch):
    # The original V260 failure used a real backslash key, unlike its old tests.
    mapping = runner.fingerprints([runner.ROOT / runner.DESIGN])
    expected = next(iter(mapping.values()))
    monkeypatch.setattr(runner, "fingerprint", lambda path: {"modelDesign.py": {"sha256": expected}})
    runner.require_design(Path("unused"), {"input_sha256": mapping})
    with pytest.raises(ValueError):
        runner.require_design(Path("unused"), {"input_sha256": {runner.DESIGN: "wrong"}})


def test_two_step_dependency_cannot_unlock_full_training(monkeypatch):
    monkeypatch.setattr(runner, "read", lambda path: {"training_report": report(True)})
    monkeypatch.setattr(runner, "require_audit", lambda *args: None)
    with pytest.raises(ValueError):
        runner.completed_dependencies()


def test_incomplete_queue_is_wait_not_completion(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "CONFIRM_DECISION", tmp_path / "no_confirmation.json")
    moments = iter([0., 2.])
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(moments))
    monkeypatch.setattr(runner, "completed_dependencies", lambda: pytest.fail("must not treat missing files as complete"))
    with pytest.raises(TimeoutError):
        runner.wait_dependencies(1.)

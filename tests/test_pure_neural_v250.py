import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_pure_neural_long_budget_v250 as runner
from run_pure_neural_rms_budget_v242 import command as control_command


def test_only_budget_patience_and_output_change():
    expected = control_command("eight")
    for flag, value in (("--steps", "72000"), ("--patience", "72"), ("--output-dir", runner.OUTPUT)):
        expected[expected.index(flag) + 1] = value
    assert runner.command() == expected
    assert "artifacts/pure_neural_v227/joint_low" in expected


def history(steps):
    return [{"step": step, "loss": .5, "efficiency": 74., "fairness": 55., "final": 68.3}
            for step in range(0, steps + 1, 1000)]


def test_prefix_covers_all_36k_not_just_12k():
    report, control = {"history": history(72000)}, {"history": history(36000)}
    assert runner.check_prefix(report, control)["matched"]
    report["history"][36]["final"] += .01
    assert not runner.check_prefix(report, control)["matched"]
    with pytest.raises(ValueError):
        runner.check_prefix({"history": history(12000)}, control)


def test_prefix_rejects_duplicate_or_missing_checkpoint():
    control = {"history": history(36000)}
    report = {"history": history(72000)}
    bad = copy.deepcopy(report)
    bad["history"][20]["step"] = 19000
    with pytest.raises(ValueError):
        runner.check_prefix(bad, control)


def test_full_report_requires_complete_same_protocol():
    report = {"requested_steps": 72000, "batch_size": 100, "seed": 15240, "learning_rate": 1e-5,
              "loss_kind": "rms_score", "score_temperature": .5, "score_bce_weight": .05,
              "score_fairness_weight": .3, "quantile_bandwidth": .025,
              "train_components": ["transmitter", "receiver"], "validation_samples": 2000,
              "baseline_expert_map": [0, 0, 1, 1, 1, 1, 1, 1], "gpu_memory_fraction": .4,
              "gpu_peak_allocated_bytes": 1, "history": history(72000)}
    runner.require_report(report, 72000)
    for changes in ({"history": history(71000)}, {"loss_kind": "soft_score"}, {"batch_size": 400}):
        with pytest.raises(ValueError):
            runner.require_report({**report, **changes}, 72000)

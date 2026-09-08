import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_pure_neural_rms_eight_v241 as runner


def test_combination_retains_12k_factorial_budget_and_eight_expert_mapping():
    args = runner.command()
    assert runner.PLAN["mapping"] == [0, 0, 1, 1, 1, 1, 1, 1]
    assert args[2] == "scripts/train_pure_neural_rms_v239.py"
    for flag, expected in (("--steps", "12000"), ("--patience", "12"), ("--loss-kind", "rms_score"),
                           ("--seed", "15240"), ("--learning-rate", "1e-5"), ("--batch-size", "100"),
                           ("--model-design", "research/pure_neural_v230/modelDesign.py")):
        assert args[args.index(flag) + 1] == expected


def test_dependency_requires_36k_and_exact_registered_audit_protocol():
    report = {
        "label": "v240_two_36k", "per_seed": [{}, {}, {}],
        "protocol": {"split_seed": 1176, "test_offset": 2000, "samples": 2000,
                     "noise_seeds": [22701, 22702, 22703]},
        "training_report": {"requested_steps": 36000, "history": [{"step": 0}, {"step": 36000}]},
    }
    runner.require_dependency(report)
    report["training_report"]["history"][-1]["step"] = 12000
    with pytest.raises(ValueError, match="36k"):
        runner.require_dependency(report)
    report["training_report"]["history"][-1]["step"] = 36000
    report["protocol"]["split_seed"] = 1177
    with pytest.raises(ValueError, match="36k"):
        runner.require_dependency(report)

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_pure_neural_budget_v240 as runner


def test_nested_budget_keeps_protocol_except_steps_and_early_stop_budget():
    for arm in ("two", "eight"):
        args = runner.command(arm)
        for flag, expected in (("--steps", "36000"), ("--patience", "36"), ("--learning-rate", "1e-5"),
                               ("--seed", "15240"), ("--batch-size", "100"), ("--loss-kind", "soft_score"),
                               ("--validation-samples", "2000"), ("--validate-every", "1000")):
            assert args[args.index(flag) + 1] == expected
        assert runner.plan(arm)["parent"] == "artifacts/pure_neural_v227/joint_low"
    assert runner.plan("eight")["mapping"] == [0, 0, 1, 1, 1, 1, 1, 1]


def test_prefix_check_detects_non_nested_trajectory():
    history = [{"step": step, "loss": .5, "efficiency": 74., "fairness": 55., "final": 68.3}
               for step in range(0, 13000, 1000)]
    report = {"history": [dict(row) for row in history] + [{"step": 36000}]}
    control = {"history": history}
    assert runner.check_prefix(report, control)["matched"]
    report["history"][3]["final"] += .01
    assert not runner.check_prefix(report, control)["matched"]
    with pytest.raises(ValueError, match="checkpoints"):
        runner.check_prefix({"history": history[:-1]}, control)

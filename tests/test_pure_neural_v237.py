import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v237_runner", ROOT / "scripts/run_pure_neural_lr_v237.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def argument(command, flag):
    return command[command.index(flag) + 1]


def test_lr_arms_preserve_control_protocol_and_separate_outputs():
    two, eight = (runner.experiment(arm) for arm in ("two", "eight"))
    assert two["mapping"] == [0, 1]
    assert eight["mapping"] == [0, 0, 1, 1, 1, 1, 1, 1]
    assert Path(two["output"]).parent != Path(eight["output"]).parent
    for plan in (two, eight):
        assert (ROOT / plan["design"]).is_file()
        command = runner.train_command(plan)
        assert argument(command, "--learning-rate") == "5e-5"
        assert argument(command, "--steps") == "12000"
        assert argument(command, "--seed") == "15240"
        assert argument(command, "--batch-size") == "100"
        assert argument(command, "--loss-kind") == "soft_score"
        assert argument(command, "--tail-fraction") == "0.1"
        assert argument(command, "--score-temperature") == "0.5"
        assert argument(command, "--baseline-dir") == "artifacts/pure_neural_v227/joint_low"
        i = command.index("--train-components")
        assert command[i + 1:i + 3] == ["transmitter", "receiver"]
        assert "--optimize-expert-index" not in command


def test_incomplete_dependency_is_not_a_completed_experiment():
    runner.require_complete({"requested_steps": 12000, "history": [{"step": 0}, {"step": 12000}]})
    for report in ({}, {"requested_steps": 12000, "history": []},
                   {"requested_steps": 12000, "history": [{"step": 1000}]}):
        with pytest.raises(ValueError):
            runner.require_complete(report)
    with pytest.raises(ValueError):
        runner.experiment("unknown")

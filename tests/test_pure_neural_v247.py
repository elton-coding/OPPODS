import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_pure_neural_batch_lr_v247 import DEPENDENCY, command, factorial_command, require_dependency
from run_pure_neural_global_batch_v246 import command as reference_command


@pytest.mark.parametrize("probe", [False, True])
def test_only_lr_and_output_change_from_v246(probe):
    actual = command(probe=probe)
    reference = reference_command(probe=probe)
    assert actual[actual.index("--learning-rate") + 1] == "5e-5"
    assert "v247" in actual[actual.index("--output-dir") + 1]
    for flag in ("--learning-rate", "--output-dir"):
        actual[actual.index(flag) + 1] = reference[reference.index(flag) + 1]
    assert actual == reference


def test_dependency_requires_actual_global_batch_budget_and_fixed_protocol():
    training = {"requested_steps": 3000, "history": [{"step": 3000}], "batch_size": 400,
                "microbatch_size": 100, "validation_batch_size": 100, "global_loss_ue_count": 800,
                "training_channel_draws": 1200000, "loss_kind": "soft_score", "score_bce_weight": .05,
                "gpu_peak_allocated_bytes": 1, "learning_rate": 1e-5}
    report = {"training_report": training, "label": DEPENDENCY, "per_seed": [1, 2, 3],
              "protocol": {"split_seed": 1176, "test_offset": 2000,
                           "samples": 2000, "noise_seeds": [22701, 22702, 22703]}}
    require_dependency(report)
    with pytest.raises(ValueError):
        require_dependency({**report, "protocol": {}})
    with pytest.raises(ValueError):
        require_dependency({**report, "training_report": {**training, "requested_steps": 12000}})
    with pytest.raises(ValueError):
        require_dependency({**report, "training_report": {**training, "learning_rate": 5e-5}})


def test_factorial_uses_the_four_matched_sample_budget_arms():
    args = factorial_command()
    for flag, label in (("--c", "v230_control"), ("--a", "v237_two_lr5"),
                        ("--b", "v246_global_batch400"), ("--ab", "v247_batch400_lr5")):
        index = args.index(flag)
        assert all(label in value for value in args[index + 1:index + 4])

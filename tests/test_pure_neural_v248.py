import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_pure_neural_temperature_v248 as runner
from run_pure_neural_lr_v237 import train_command
from train_pure_neural_snr_experts import score_aligned_loss


def test_temperature_is_only_command_change():
    baseline = train_command(runner.PLAN)
    actual = runner.command()
    index = baseline.index("--score-temperature") + 1
    assert actual[index] == "1.0"
    actual[index] = baseline[index]
    assert actual == baseline
    probe = runner.command(probe=True)
    assert probe[probe.index("--batch-size") + 1] == "100"
    assert probe[probe.index("--steps") + 1] == "2"
    assert runner.PLAN["parent"] == "artifacts/pure_neural_v227/joint_low"


def test_report_rejects_incomplete_and_wrong_temperature():
    report = {"requested_steps": 12000, "history": [{"step": 12000}], "score_temperature": 1.,
              "score_bce_weight": .05, "score_fairness_weight": .3, "quantile_bandwidth": .025,
              "loss_kind": "soft_score", "batch_size": 100, "learning_rate": 1e-5,
              "seed": 15240, "baseline_expert_map": [0, 1], "gpu_peak_allocated_bytes": 1,
              "train_components": ["transmitter", "receiver"], "validation_samples": 2000}
    runner.require_report(report)
    for change in ({"history": [{"step": 11000}]}, {"score_temperature": .5}, {"batch_size": 400}):
        with pytest.raises(ValueError):
            runner.require_report({**report, **change})


def test_eight_expert_factorial_maps_same_architecture():
    paired, factorial = runner.predecessor_commands()
    assert paired[paired.index("--baseline") + 1:paired.index("--candidate")] == runner.score_paths("v240_eight_36k")
    for arm, label in (("c", "v230_eight"), ("a", "v241_eight_rms"),
                       ("b", "v240_eight_36k"), ("ab", "v242_eight_rms_36k")):
        index = factorial.index(f"--{arm}") + 1
        assert factorial[index:index + 3] == runner.score_paths(label)


def test_t1_proxy_matches_definition_and_has_finite_gradients():
    torch.manual_seed(248)
    logits = torch.randn(100, 2, 16, dtype=torch.float64, requires_grad=True)
    bits = torch.randint(0, 2, logits.shape).double()
    loss = score_aligned_loss(logits, bits, loss_kind="soft_score", margin=.5, tail_weight=0.,
                              tail_fraction=.1, score_temperature=1., quantile_bandwidth=.025,
                              score_bce_weight=.05, score_fairness_weight=.3)
    scores = torch.sigmoid((2 * bits - 1) * logits).mean(-1).flatten()
    ranks = torch.arange(scores.numel(), dtype=logits.dtype)
    weights = torch.softmax(-.5 * ((ranks - .1 * (scores.numel() - 1)) / (.025 * scores.numel())) ** 2, dim=0)
    expected = -.7 * scores.mean() - .3 * (scores.sort().values * weights).sum()
    expected += .05 * torch.nn.functional.binary_cross_entropy_with_logits(logits, bits)
    torch.testing.assert_close(loss, expected, atol=1e-12, rtol=1e-12)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0

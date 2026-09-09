import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_pure_neural_bce_v245 import ARMS, command, plan, require_arm
from run_pure_neural_lr_v237 import train_command
from train_pure_neural_snr_experts import score_aligned_loss


@pytest.mark.parametrize("arm", ARMS)
def test_only_bce_differs_from_control_command(arm):
    baseline = train_command(plan(arm))
    candidate = command(arm)
    index = baseline.index("--score-bce-weight") + 1
    assert candidate[index] == ARMS[arm]
    candidate[index] = baseline[index]
    assert candidate == baseline
    probe = command(arm, probe=True)
    assert probe[probe.index("--steps") + 1] == "2"
    assert probe[probe.index("--batch-size") + 1] == "100"
    assert probe[probe.index("--output-dir") + 1] != baseline[baseline.index("--output-dir") + 1]


@pytest.mark.parametrize("weight", [0., .01, .05])
def test_score_proxy_remains_differentiable_without_auxiliary_bce(weight):
    torch.manual_seed(245)
    logits = torch.randn(100, 2, 16, dtype=torch.float64, requires_grad=True)
    bits = torch.randint(0, 2, logits.shape).double()
    options = {"loss_kind": "soft_score", "margin": .5, "tail_weight": 0., "tail_fraction": .1,
               "score_temperature": .5, "quantile_bandwidth": .025, "score_fairness_weight": .3}
    proxy = score_aligned_loss(logits, bits, score_bce_weight=0., **options)
    loss = score_aligned_loss(logits, bits, score_bce_weight=weight, **options)
    expected = weight * torch.nn.functional.binary_cross_entropy_with_logits(logits, bits)
    torch.testing.assert_close(loss - proxy, expected, atol=1e-12, rtol=1e-12)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_reject_incomplete_or_wrong_weight():
    report = {"requested_steps": 12000, "history": [{"step": 12000}], "score_bce_weight": 0.,
              "loss_kind": "soft_score", "batch_size": 100, "baseline_expert_map": [0, 1],
              "gpu_peak_allocated_bytes": 1}
    require_arm(report, "zero")
    with pytest.raises(ValueError):
        require_arm(report, "low")
    with pytest.raises(ValueError):
        require_arm({**report, "history": [{"step": 11000}]}, "zero")

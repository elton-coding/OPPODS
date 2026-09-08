import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_pure_neural_rms_v239 as rms_trainer

OPTIONS = {"margin": .5, "tail_weight": 0., "tail_fraction": .1, "score_temperature": .5,
           "quantile_bandwidth": .025, "score_fairness_weight": .3}


def loss(logits, bits, bce=0.):
    return rms_trainer.rms_score_loss(logits, bits, loss_kind="rms_score", score_bce_weight=bce, **OPTIONS)


def test_proxy_is_invariant_to_each_ue_positive_scale_with_radial_zero_gradient():
    torch.manual_seed(239)
    logits = torch.randn(8, 2, 64, dtype=torch.float64, requires_grad=True)
    bits = torch.randint(0, 2, logits.shape).double()
    scales = torch.rand(8, 2, 1, dtype=torch.float64) * 10 + .1
    reference = loss(logits, bits)
    torch.testing.assert_close(reference, loss(logits * scales, bits), atol=1e-12, rtol=1e-12)
    reference.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0
    radial = (logits.grad * logits).sum(-1)
    torch.testing.assert_close(radial, torch.zeros_like(radial), atol=1e-12, rtol=0)


def test_raw_bce_is_unchanged_and_zero_outputs_have_finite_gradients():
    torch.manual_seed(239)
    logits = torch.randn(4, 2, 32, dtype=torch.float64)
    bits = torch.randint(0, 2, logits.shape).double()
    expected = .05 * torch.nn.functional.binary_cross_entropy_with_logits(logits, bits)
    torch.testing.assert_close(loss(logits, bits, .05) - loss(logits, bits), expected)
    zeros = torch.zeros_like(logits, requires_grad=True)
    loss(zeros, bits, .05).backward()
    assert torch.isfinite(zeros.grad).all()
    assert zeros.grad.abs().sum() > 0


def test_no_other_loss_or_shared_trainer_state_changes():
    original = rms_trainer.trainer.score_aligned_loss
    assert original is rms_trainer.BASE_LOSS
    logits = torch.randn(4, 2, 32)
    bits = torch.randint(0, 2, logits.shape).float()
    candidate = rms_trainer.rms_score_loss(logits, bits, loss_kind="soft_score", **OPTIONS)
    reference = original(logits, bits, loss_kind="soft_score", **OPTIONS)
    torch.testing.assert_close(candidate, reference, atol=0, rtol=0)
    with pytest.raises(ValueError, match="full-length"):
        loss(logits[..., :16], bits)


def test_cli_reports_actual_loss_and_restores_argv(monkeypatch):
    argv = ["train_pure_neural_rms_v239.py", "--stage", "calibrate", "--loss-kind", "rms_score"]
    monkeypatch.setattr(sys, "argv", argv)
    args = rms_trainer.parse_args()
    assert args.loss_kind == "rms_score"
    assert sys.argv is argv
    assert rms_trainer.trainer.parse_args is rms_trainer.BASE_PARSE


def test_entry_restores_functions_even_when_training_fails(monkeypatch):
    def fail():
        assert rms_trainer.trainer.score_aligned_loss is rms_trainer.rms_score_loss
        raise RuntimeError("deliberate failure")
    monkeypatch.setattr(rms_trainer.trainer, "main", fail)
    with pytest.raises(RuntimeError, match="deliberate failure"):
        rms_trainer.main()
    assert rms_trainer.trainer.parse_args is rms_trainer.BASE_PARSE
    assert rms_trainer.trainer.score_aligned_loss is rms_trainer.BASE_LOSS


def test_formal_command_uses_new_objective_without_changing_control_budget():
    from run_pure_neural_rms_v239 import command
    args = command()
    assert args[2] == "scripts/train_pure_neural_rms_v239.py"
    for flag, expected in (("--loss-kind", "rms_score"), ("--steps", "12000"),
                           ("--learning-rate", "1e-5"), ("--batch-size", "100"), ("--seed", "15240"),
                           ("--model-design", "research/pure_neural_v227/modelDesign.py"),
                           ("--output-dir", "artifacts/pure_neural_v239/rms_score")):
        assert args[args.index(flag) + 1] == expected

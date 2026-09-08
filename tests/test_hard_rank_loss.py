import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "hard_rank_trainer", Path(__file__).resolve().parents[1] / "scripts/train_pure_neural_snr_experts.py",
)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


def test_resuming_an_incomplete_checkpoint_fails(tmp_path):
    link = SimpleNamespace(encoder=torch.nn.Linear(2, 2))
    torch.save({}, tmp_path / "encoder.pth")
    with pytest.raises(RuntimeError, match="Missing key"):
        trainer.PureNeuralLink.load_submission(link, tmp_path)


@pytest.mark.parametrize("output_bits", [12, 16])
def test_forward_matches_official_hard_score_including_ties_and_missing_bits(output_bits):
    torch.manual_seed(228)
    bits = torch.randint(0, 2, (10, 2, 16)).float()
    logits = torch.randn(10, 2, output_bits)
    logits[..., 0] = 0  # The official decision is >= 0, not signed_logit > 0.
    logits.requires_grad_()
    loss = trainer.score_aligned_loss(
        logits, bits, loss_kind="hard_rank_score", margin=0.5, tail_weight=0,
        tail_fraction=0.1, score_bce_weight=0,
    )
    hard = ((logits.detach() >= 0) == (bits[..., :output_bits] >= .5)).sum(-1)
    score = (hard.numpy() + .5 * (16 - output_bits)) / 16
    expected = -(.7 * score.mean() + .3 * np.percentile(score, 10))
    assert float(loss.detach()) == pytest.approx(expected, abs=1e-7)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    signed_grad = logits.grad * (2 * bits[..., :output_bits] - 1)
    assert torch.all(signed_grad < 0)


def test_hard_rank_gradient_differs_when_logit_confidence_misranks_links():
    # Two wrong but very confident bits can give a better soft score than six
    # correct but low-confidence bits. Hard ranking must not use that ordering.
    bits = torch.ones(10, 2, 8)
    logits = torch.full_like(bits, .02)
    logits[:5, :, :] = 4
    logits[:5, :, :2] = -4
    gradients = []
    for kind in ("soft_score", "hard_rank_score"):
        x = logits.clone().requires_grad_()
        trainer.score_aligned_loss(
            x, bits, loss_kind=kind, margin=.5, tail_weight=0,
            tail_fraction=.1, score_bce_weight=0,
        ).backward()
        gradients.append(x.grad)
    assert not torch.allclose(*gradients)

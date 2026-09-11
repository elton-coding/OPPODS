import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import diagnose_shared_gradients_v264 as d


def test_global_loss_and_parameter_gradients_decompose():
    torch.manual_seed(264)
    parameter = torch.randn(5, 1152, requires_grad=True)
    features = torch.randn(100, 2, 5)
    logits = features @ parameter
    bits = torch.randint(0, 2, logits.shape).float()
    snr = torch.rand(100, 2) * 40 - 20
    parts = d.contributions(logits, bits)
    actual = d.rms_score_loss(logits, bits, loss_kind="rms_score", margin=.5, tail_weight=0., tail_fraction=.1,
                              score_temperature=.5, quantile_bandwidth=.025, score_bce_weight=.05,
                              score_fairness_weight=.3)
    torch.testing.assert_close(parts.sum(), actual)
    masks = d.masks(snr)
    assert torch.all(sum(v.int() for v in masks.values()) == 1)
    gradients = [d.gradient_vector(parts[mask].sum(), [parameter]) for mask in masks.values()]
    torch.testing.assert_close(sum(gradients), d.gradient_vector(actual, [parameter]), atol=1e-7, rtol=1e-5)


def test_conditions_and_zero_gradient_cosine():
    snr = torch.tensor([[-17., -2.], [-17., 5.], [6., -18.]])
    masks = d.masks(snr)
    assert masks["weak_both_negative"].tolist() == [True, False, False, False, False, False]
    assert masks["strong_with_weak_partner"].tolist() == [False, False, False, True, True, False]
    assert d.cosine(torch.zeros(2), torch.ones(2)) is None

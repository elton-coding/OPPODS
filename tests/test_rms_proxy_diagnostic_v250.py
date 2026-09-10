import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from diagnose_rms_proxy_v250 import analyze, average_ranks, rank_correlation, rank_weights


def test_average_ranks_handle_ties_without_order_bias():
    ranks = average_ranks(torch.tensor([3.,1.,1.,2.]))
    torch.testing.assert_close(ranks,torch.tensor([3.,.5,.5,2.],dtype=torch.float64),atol=0,rtol=0)
    assert rank_correlation(torch.ones(4),torch.arange(4).float()) is None
    assert rank_correlation(torch.arange(4).float(),-torch.arange(4).float()) == pytest.approx(-1, abs=1e-14)


def test_rank_weights_match_original_sorted_gaussian_and_are_permutation_equivariant():
    scores = torch.tensor([.9,.4,.8,.2,.6])
    weights = rank_weights(scores)
    ranks = torch.arange(5).float()
    expected = torch.softmax(-.5*((ranks-.4)/1.).square(),0)
    torch.testing.assert_close(weights[torch.sort(scores).indices],expected,atol=0,rtol=0)
    order = torch.tensor([2,0,4,1,3])
    torch.testing.assert_close(rank_weights(scores[order]),weights[order],atol=0,rtol=0)
    torch.testing.assert_close(weights.sum(),torch.tensor(1.))


@pytest.mark.parametrize("temperature", [.5,.25])
def test_analyze_reconstructs_actual_loss_without_mutating_inputs(temperature):
    rng = torch.Generator().manual_seed(16)
    logits = torch.randn(12,2,1152,generator=rng)
    bits = torch.randint(0,2,logits.shape,generator=rng).float()
    snr = torch.linspace(-20,20,24).reshape(12,2)
    before = logits.clone()
    result = analyze(logits,bits,snr,temperature)
    torch.testing.assert_close(logits,before,atol=0,rtol=0)
    assert logits.grad is None
    assert abs(result["reconstructed_objective_difference"]) <= 2e-7
    assert 0 <= result["hard_tail_recalled_by_soft_tail"] <= 1
    assert 0 <= result["quantile_rank_weight_mass_at_hard_p10_plus_minus_one_point"] <= 1.000001
    assert result["p10_only_gradient_mass"]["absolute_gradient_sum"] > 0


def test_zero_logits_use_official_nonnegative_bit_one_and_remain_finite():
    result = analyze(torch.zeros(4,2,1152),torch.zeros(4,2,1152),torch.zeros(4,2),.5)
    assert result["hard_efficiency_training_diagnostic_only"] == 0
    assert result["hard_p10_training_diagnostic_only"] == 0
    assert result["average_rank_correlation"] is None
    assert result["total_loss_gradient_mass"]["absolute_gradient_sum"] > 0


def test_short_payload_or_nonbinary_targets_are_rejected():
    with pytest.raises(ValueError):
        analyze(torch.zeros(4,2,8),torch.zeros(4,2,8),torch.zeros(4,2),.5)
    with pytest.raises(ValueError):
        analyze(torch.zeros(4,2,1152),torch.full((4,2,1152),.5),torch.zeros(4,2),.5)

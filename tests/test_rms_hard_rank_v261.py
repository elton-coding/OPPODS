import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "scripts"))
import train_pure_neural_rms_hard_rank_v261 as entry
from probe_pure_neural_rms_hard_v261 import contract, hard_forward_reference, objective, options
from train_pure_neural_rms_v239 import rms_score_loss


@pytest.mark.parametrize("scale",[0.,1e-6,1.,1e6])
def test_full_batch_hard_forward_and_manual_gradient_contract(scale):
    rng = torch.Generator().manual_seed(9261)
    x = scale*torch.randn(100,2,1152,generator=rng)
    bits = torch.randint(0,2,x.shape,generator=rng).float()
    result = contract(x,bits)
    assert abs(result["hard_forward_difference"]) <= 2e-7
    assert result["manual_surrogate_gradient_maximum_difference"] <= 1e-7


def test_zero_threshold_matches_official_bit_one_convention():
    for bit,expected in [(0,0.),(1,-1.)]:
        x = torch.zeros(1,2,1152,requires_grad=True)
        y = torch.full_like(x,float(bit))
        loss = objective(x,y,bce=0.)
        assert float(loss.detach()) == expected
        gradient = torch.autograd.grad(loss,x)[0]
        assert torch.isfinite(gradient).all()
        assert torch.all(gradient > 0) if bit == 0 else torch.all(gradient < 0)


def reversed_soft_and_hard_rows():
    x = torch.full((1,2,1152),-.01)
    x[0,0,:900] = .01
    x[0,0,900:] = -10.
    x[0,1,:700] = 1.
    return x,torch.ones_like(x)


def test_hard_ranking_really_changes_tail_gradient_without_changing_mean_gradient():
    x,bits = reversed_soft_and_hard_rows()
    for fairness in (0.,.3):
        leaf = x.clone().requires_grad_(True)
        hard = objective(leaf,bits,fairness=fairness)
        soft = rms_score_loss(leaf,bits,loss_kind="rms_score",**options(fairness=fairness))
        hard_gradient = torch.autograd.grad(hard,leaf)[0]
        soft_gradient = torch.autograd.grad(soft,leaf)[0]
        if fairness == 0:
            torch.testing.assert_close(hard_gradient,soft_gradient,atol=1e-7,rtol=1e-5)
        else:
            assert float((hard_gradient-soft_gradient).abs().max()) > 1e-7
        torch.testing.assert_close(hard.detach(),hard_forward_reference(leaf,bits,fairness=fairness).detach(),atol=2e-7,rtol=0)


def test_score_part_stays_scale_invariant_but_raw_bce_is_not_normalized():
    x,bits = reversed_soft_and_hard_rows()
    a,b = objective(x,bits,bce=0.),objective(3*x,bits,bce=0.)
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    delta = objective(3*x,bits)-objective(x,bits)
    expected = .05*(torch.nn.functional.softplus(-(2*bits-1)*3*x).mean()-torch.nn.functional.softplus(-(2*bits-1)*x).mean())
    torch.testing.assert_close(delta,expected,atol=2e-7,rtol=0)


@pytest.mark.parametrize("kind",["rms_score","soft_score","hard_rank_score"])
def test_legacy_loss_labels_cannot_silently_select_another_objective(kind):
    x = torch.zeros(1,2,1152)
    with pytest.raises(ValueError):
        entry.rms_hard_rank_loss(x,x,loss_kind=kind,**options())


def test_short_payload_and_variable_lengths_rejected():
    with pytest.raises(ValueError):
        objective(torch.zeros(1,2,8),torch.zeros(1,2,8))
    x = torch.zeros(1,2,1152)
    settings = options()
    settings["valid_lengths"] = torch.tensor([[1152,1152]])
    with pytest.raises(ValueError):
        entry.rms_hard_rank_loss(x,x,loss_kind=entry.LOSS_KIND,**settings)


def test_parser_labels_real_loss_and_restores_argv(monkeypatch):
    argv = ["entry","--loss-kind",entry.LOSS_KIND,"--stage","calibrate"]
    monkeypatch.setattr(sys,"argv",argv)
    seen = []
    def parse():
        seen.extend(sys.argv)
        return SimpleNamespace(stage="calibrate")
    monkeypatch.setattr(entry,"BASE_PARSE",parse)
    assert entry.parse_args().loss_kind == entry.LOSS_KIND
    assert sys.argv is argv and seen[-2:] == ["--loss-kind","hard_rank_score"]


@pytest.mark.parametrize("fail",[False,True])
def test_entry_restores_original_hooks_on_success_and_exception(monkeypatch,fail):
    original = entry.trainer.parse_args,entry.trainer.score_aligned_loss
    def run():
        assert entry.trainer.parse_args is entry.parse_args
        assert entry.trainer.score_aligned_loss is entry.rms_hard_rank_loss
        if fail:
            raise RuntimeError("fixture error")
    monkeypatch.setattr(entry.trainer,"main",run)
    if fail:
        with pytest.raises(RuntimeError,match="fixture error"):
            entry.main()
    else:
        entry.main()
    assert (entry.trainer.parse_args,entry.trainer.score_aligned_loss) == original

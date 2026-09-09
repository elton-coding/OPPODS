import copy
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_pure_neural_rms_temperature_v256 as runner
from train_pure_neural_rms_v239 import rms_score_loss


def loss(logits, bits, *, temperature=.25, bce=.05):
    return rms_score_loss(logits, bits, loss_kind="rms_score", margin=.5, tail_weight=0.,
                          tail_fraction=.1, score_temperature=temperature, quantile_bandwidth=.025,
                          score_bce_weight=bce, score_fairness_weight=.3, valid_lengths=None)


@pytest.mark.parametrize("scale", [0., 1e-8, 1., 10000.])
def test_rms_temperature025_has_finite_full_payload_loss_and_gradient(scale):
    generator = torch.Generator().manual_seed(256001)
    logits = (torch.randn(100, 2, 1152, generator=generator) * scale).requires_grad_()
    bits = torch.randint(0, 2, logits.shape, generator=generator).float()
    value = loss(logits, bits)
    value.backward()
    assert torch.isfinite(value) and torch.isfinite(logits.grad).all()


def test_temperature_changes_training_gradient_but_not_hard_decisions():
    generator = torch.Generator().manual_seed(256002)
    logits = torch.randn(6, 2, 1152, generator=generator, requires_grad=True)
    bits = torch.randint(0, 2, logits.shape, generator=generator).float()
    before = logits.detach().clone()
    gradient025 = torch.autograd.grad(loss(logits, bits), logits)[0]
    gradient05 = torch.autograd.grad(loss(logits, bits, temperature=.5), logits)[0]
    assert not torch.equal(gradient025, gradient05)
    assert torch.equal(before, logits)
    assert torch.equal(before >= 0, logits >= 0)


def test_rms_proxy_remains_scale_invariant_and_raw_bce_weight_unchanged():
    generator = torch.Generator().manual_seed(256003)
    logits = torch.randn(6, 2, 1152, generator=generator, dtype=torch.float64)
    bits = torch.randint(0, 2, logits.shape, generator=generator).double()
    proxy = loss(logits, bits, bce=0.)
    torch.testing.assert_close(proxy, loss(logits * 7., bits, bce=0.), atol=1e-12, rtol=1e-12)
    raw_bce = torch.nn.functional.softplus(-(2. * bits - 1.) * logits).mean()
    torch.testing.assert_close(loss(logits, bits) - proxy, .05 * raw_bce, atol=1e-12, rtol=1e-12)


def test_only_temperature_and_output_differ_from_full72k_constant_control():
    expected = runner.control_command()
    expected[expected.index("--score-temperature") + 1] = "0.25"
    expected[expected.index("--output-dir") + 1] = runner.OUTPUT
    assert runner.command() == expected
    assert runner.command()[2] == "scripts/train_pure_neural_rms_v239.py"
    probe = runner.command(probe=True)
    assert probe[probe.index("--steps") + 1] == "2"
    assert probe[probe.index("--batch-size") + 1] == "100"
    assert probe[probe.index("--validation-samples") + 1] == "2000"


def test_full_report_guard_rejects_partial_wrong_temperature_and_cosine_combination():
    original = json.loads((ROOT / "benchmarks/v250_eight_rms_72k_audit_offset2000.json").read_text(encoding="utf-8"))["training_report"]
    report = copy.deepcopy(original)
    with pytest.raises(ValueError):
        runner.require_result(report)
    report["score_temperature"] = .25
    runner.require_result(report)  # Synthetic schema fixture, not a V256 training claim.
    assert original["score_temperature"] == .5
    report["learning_rate_schedule"] = {}
    with pytest.raises(ValueError):
        runner.require_result(report)
    del report["learning_rate_schedule"]
    report["history"].pop()
    with pytest.raises(ValueError):
        runner.require_result(report)


def test_wait_does_not_accept_missing_or_partial_dependency(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    with pytest.raises(TimeoutError):
        runner.wait_dependency(0)
    path = tmp_path / f"benchmarks/{runner.DEPENDENCY}_audit_offset2000.json"
    path.parent.mkdir()
    path.write_text("{", encoding="utf-8")
    with pytest.raises(TimeoutError):
        runner.wait_dependency(0)
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        runner.wait_dependency(0)

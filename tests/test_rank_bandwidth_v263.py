import copy
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_pure_neural_rank_bandwidth_v263 as r
from train_pure_neural_rms_v239 import rms_score_loss


def loss(x, bits, bandwidth=.05, bce=.05):
    return rms_score_loss(x, bits, loss_kind="rms_score", margin=.5, tail_weight=0.,
                          tail_fraction=.1, score_temperature=.5, quantile_bandwidth=bandwidth,
                          score_bce_weight=bce, score_fairness_weight=.3)


@pytest.mark.parametrize("scale", [0., 1e-8, 1., 10000.])
def test_full_batch_finite(scale):
    g = torch.Generator().manual_seed(263)
    x = (torch.randn(100, 2, 1152, generator=g) * scale).requires_grad_()
    bits = torch.randint(0, 2, x.shape, generator=g).float()
    value = loss(x, bits)
    value.backward()
    assert torch.isfinite(value) and torch.isfinite(x.grad).all()


def test_manual_kernel_and_gradient_change():
    g = torch.Generator().manual_seed(26301)
    x = torch.randn(100, 2, 1152, generator=g, dtype=torch.float64, requires_grad=True)
    bits = torch.randint(0, 2, x.shape, generator=g).double()
    z = x / x.square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
    scores = torch.sigmoid((2 * bits - 1) * z / .5).mean(-1).flatten()
    ranks = torch.arange(200, dtype=x.dtype)
    weights = torch.softmax(-.5 * ((ranks - 19.9) / 10).square(), dim=0)
    expected = -.7 * scores.mean() - .3 * (scores.sort().values * weights).sum()
    raw = torch.nn.functional.softplus(-(2 * bits - 1) * x).mean()
    torch.testing.assert_close(loss(x, bits), expected + .05 * raw)
    grad = torch.autograd.grad(loss(x, bits), x)[0]
    old = torch.autograd.grad(loss(x, bits, .025), x)[0]
    assert not torch.equal(grad, old)
    torch.testing.assert_close(loss(x * 7, bits, bce=0), loss(x, bits, bce=0))


def fixture():
    a = json.loads((ROOT / f"benchmarks/{r.CONTROL}_audit_offset2000.json").read_text(encoding="utf-8"))
    result = copy.deepcopy(a["training_report"])
    result["quantile_bandwidth"] = .05
    return result


def test_only_bandwidth_and_output_changed():
    expected = r.control_command()
    expected[expected.index("--quantile-bandwidth") + 1] = "0.05"
    expected[expected.index("--output-dir") + 1] = r.OUTPUT
    assert r.command() == expected
    probe = r.command(probe=True)
    assert probe[probe.index("--steps") + 1] == "2"
    assert probe[probe.index("--batch-size") + 1] == "100"


@pytest.mark.parametrize("key,value", [("quantile_bandwidth", .025), ("score_bce_weight", 0.),
    ("score_fairness_weight", .5), ("score_temperature", .25), ("trainable_parameters", 73547840),
    ("requested_steps", 71000), ("learning_rate_schedule", {}), ("microbatch_size", 25)])
def test_guard_rejects_extra_factors(key, value):
    report = fixture()
    r.require_result(report)  # Synthetic schema only, not a training result.
    report[key] = value
    with pytest.raises(ValueError):
        r.require_result(report)


def test_partial_and_nonfinite_rejected():
    report = fixture()
    report["history"].pop()
    with pytest.raises(ValueError):
        r.require_result(report)
    report = fixture()
    report["history"][1]["final"] = float("nan")
    with pytest.raises(ValueError):
        r.require_result(report)


def test_selection_gate():
    comparison = {"per_seed": [{"noise_seed": seed, "delta": {"final": v}}
                               for seed, v in zip(r.SEEDS, [.01, .02, .03], strict=True)],
                  "paired_delta_95_percentile_interval": [.001, .05]}
    assert r.decision(comparison)["eligible_for_new_confirmation"]
    comparison["per_seed"][0]["delta"]["final"] = -.001
    assert not r.decision(comparison)["eligible_for_new_confirmation"]
    comparison["per_seed"][0]["delta"]["final"] = .01
    comparison["paired_delta_95_percentile_interval"][0] = -.001
    assert not r.decision(comparison)["eligible_for_new_confirmation"]


def test_actual_control_metrics_and_tampering():
    audit = json.loads((ROOT / f"benchmarks/{r.CONTROL}_audit_offset2000.json").read_text(encoding="utf-8"))
    cached = r.checked_caches([r.CONTROL], offset=2000, seeds=r.SEEDS)[r.CONTROL]
    r.require_metrics(audit, cached)
    audit["per_seed"][0]["final"] += .1
    with pytest.raises(ValueError):
        r.require_metrics(audit, cached)


def test_existing_output_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(r, "ROOT", tmp_path)
    (tmp_path / r.OUTPUT).mkdir(parents=True)
    with pytest.raises(FileExistsError):
        r.main()

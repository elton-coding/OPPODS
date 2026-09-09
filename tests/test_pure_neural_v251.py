import copy
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_pure_neural_fairness_v251 as runner
from run_pure_neural_rms_budget_v242 import command as control_command
from train_pure_neural_rms_v239 import rms_score_loss


def test_only_surrogate_weight_and_output_change():
    expected = control_command("eight")
    for flag, value in (("--score-fairness-weight", "0.5"), ("--output-dir", runner.OUTPUT)):
        expected[expected.index(flag) + 1] = value
    assert runner.command() == expected


def test_candidate_budget_and_objective_are_checked_without_mutation():
    path = runner.ROOT / f"benchmarks/{runner.CONTROL}_audit_offset2000.json"
    report = json.loads(path.read_text(encoding="utf-8"))["training_report"]
    with pytest.raises(ValueError):
        runner.require_candidate(report)
    candidate = {**report, "score_fairness_weight": .5}
    snapshot = copy.deepcopy(candidate)
    runner.require_candidate(candidate)
    assert candidate == snapshot
    for changes in ({"requested_steps": 12000}, {"loss_kind": "soft_score"}, {"score_bce_weight": 0.}):
        with pytest.raises(ValueError):
            runner.require_candidate({**candidate, **changes})


def test_initial_validation_must_match():
    report = {"history": [{"loss": .5, "efficiency": 74., "fairness": 55., "final": 68.3}]}
    assert max(runner.check_initial(report, report).values()) == 0
    other = copy.deepcopy(report)
    other["history"][0]["fairness"] += .01
    with pytest.raises(ValueError):
        runner.check_initial(other, report)


def test_fairness_weight_really_changes_rms_gradients_and_is_finite():
    generator = torch.Generator().manual_seed(15251)
    logits = torch.randn(10, 2, 32, generator=generator, requires_grad=True)
    bits = torch.randint(0, 2, logits.shape, generator=generator).float()
    options = {"loss_kind": "rms_score", "margin": 1., "tail_weight": .3, "tail_fraction": .1,
               "score_temperature": .5, "quantile_bandwidth": .025, "score_bce_weight": .05}
    first = rms_score_loss(logits, bits, score_fairness_weight=.3, **options)
    second = rms_score_loss(logits, bits, score_fairness_weight=.5, **options)
    grad_first = torch.autograd.grad(first, logits)[0]
    grad_second = torch.autograd.grad(second, logits)[0]
    assert torch.isfinite(second) and torch.isfinite(grad_second).all()
    assert not torch.allclose(grad_first, grad_second)

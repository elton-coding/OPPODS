import copy
import sys
from itertools import pairwise
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_pure_neural_cosine_v255 as cosine


def test_registered_rate_boundaries_and_monotonic_decay():
    assert cosine.learning_rate(1) == cosine.learning_rate(36000) == 1e-5
    assert cosine.learning_rate(36001) < 1e-5
    assert cosine.learning_rate(72000) == pytest.approx(1e-6)
    values = [cosine.learning_rate(i) for i in range(36000, 72001)]
    assert all(a >= b >= 1e-6 for a, b in pairwise(values))


@pytest.mark.parametrize("step", [0, -1, 72001, 1.5])
def test_outside_registered_schedule_rejected(step):
    with pytest.raises(ValueError):
        cosine.learning_rate(step)


def test_hooks_preserve_adam_state_and_rng_in_constant_prefix():
    a, b = (torch.nn.Parameter(torch.tensor([.1, -.2])) for _ in range(2))
    oa, ob = torch.optim.Adam([a], lr=1e-5), torch.optim.Adam([b], lr=1e-5)
    recorder = cosine.RateRecorder(ob)
    state = torch.random.get_rng_state().clone()
    for step in range(1, 21):
        a.grad = torch.tensor([step * .01, -.1])
        b.grad = a.grad.clone()
        oa.step()
        ob.step()
        assert torch.equal(a, b)
        for key in oa.state[a]:
            assert torch.equal(oa.state[a][key], ob.state[b][key])
    assert torch.equal(state, torch.random.get_rng_state())
    cosine.require_schedule(recorder.evidence(), 20)


def test_decay_is_applied_before_actual_update_without_resetting_adam():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.Adam([parameter], lr=1e-5)
    recorder = cosine.RateRecorder(optimizer)
    parameter.grad = torch.ones(1)
    optimizer.step()
    moment = optimizer.state[parameter]["exp_avg"]
    recorder.completed = 71999  # A targeted hook boundary test, not full trajectory evidence.
    optimizer.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-6)
    assert optimizer.state[parameter]["exp_avg"] is moment
    assert optimizer.state[parameter]["step"].item() == 2
    with pytest.raises(ValueError):
        cosine.require_schedule(recorder.evidence(), 72000)


def test_partial_update_or_mutated_rate_cannot_be_certified():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.Adam([parameter], lr=1e-5)
    recorder = cosine.RateRecorder(optimizer)
    recorder.before(optimizer, (), {})
    with pytest.raises(RuntimeError):
        recorder.evidence()
    optimizer.param_groups[0]["lr"] = 2e-5
    with pytest.raises(RuntimeError):
        recorder.after(optimizer, (), {})


def test_evidence_digest_and_update_count_are_mandatory():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.Adam([parameter], lr=1e-5)
    recorder = cosine.RateRecorder(optimizer)
    parameter.grad = torch.ones(1)
    optimizer.step()
    evidence = recorder.evidence()
    cosine.require_schedule(evidence, 1)
    changed = copy.deepcopy(evidence)
    changed["applied_rates_float64_le_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        cosine.require_schedule(changed, 1)
    with pytest.raises(ValueError):
        cosine.require_schedule(evidence, 2)

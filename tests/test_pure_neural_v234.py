import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load(version):
    spec = importlib.util.spec_from_file_location(version, ROOT / f"research/pure_neural_{version}/modelDesign.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_feedback_gating_preserves_parent_and_has_gradient():
    torch.set_num_threads(2)
    torch.manual_seed(234)
    parent = load("v227").TransmitterCore()
    candidate = load("v234").TransmitterCore()
    candidate.load_parent_state(parent.state_dict())
    bits = [torch.randint(0, 2, (2, 1152)).float() for _ in range(2)]
    feedback = [torch.randn(2, 96, dtype=torch.complex64) for _ in range(2)]
    snr = torch.tensor([[-18., 10.], [15., 16.]])
    signal, control = candidate(bits, feedback, snr)
    original_signal, original_control = parent(bits, feedback, snr)
    torch.testing.assert_close(signal, original_signal, atol=0, rtol=0)
    torch.testing.assert_close(control, original_control, atol=0, rtol=0)
    signal.abs().square().mean().backward()
    for gate in candidate._feedback_gate:
        assert gate[-1].weight.grad.abs().sum() > 0
    broken = dict(parent.state_dict())
    del broken["_embed.weight"]
    with pytest.raises(ValueError, match="invalid parent"):
        candidate.load_parent_state(broken)

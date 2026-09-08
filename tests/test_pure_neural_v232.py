import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_zero_context_gate_preserves_parent_and_gets_gradient():
    torch.set_num_threads(2)
    torch.manual_seed(232)
    old = load(ROOT / "research/pure_neural_v227/modelDesign.py").ReceiverCore()
    new = load(ROOT / "research/pure_neural_v232/modelDesign.py").ReceiverCore()
    new.load_parent_state(old.state_dict())
    y = torch.randn(2, 2, 144, dtype=torch.complex64)
    h = torch.randn(2, 2, 16, 144, dtype=torch.complex64)
    control, snr = torch.zeros(2, 5), torch.tensor([-15., 12.])
    expected = old(y, h, control, snr)
    actual = new(y, h, control, snr)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.square().mean().backward()
    assert new._context[-1].weight.grad.abs().sum() > 0
    state = old.state_dict()
    del state["_embed.weight"]
    with pytest.raises(ValueError, match="invalid parent"):
        new.load_parent_state(state)

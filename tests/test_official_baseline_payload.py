from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import torch

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_parameterization_loads_official_weights() -> None:
    candidate = _load("official_payload_default", ROOT / "research/official_baseline_payload/modelDesign.py")
    encoder = candidate.Encoder()
    transmitter = candidate.Transmitter()
    receiver = candidate.Receiver()
    encoder.load_state_dict(torch.load(ROOT / "ziliao/modelSubmit/encoder.pth", weights_only=True))
    transmitter.load_state_dict(torch.load(ROOT / "ziliao/modelSubmit/transmitter.pth", weights_only=True))
    receiver.load_state_dict(torch.load(ROOT / "ziliao/modelSubmit/receiver.pth", weights_only=True))


def test_bits_per_re_and_payload_control_only_the_expected_shapes() -> None:
    candidate = _load("official_payload_shape", ROOT / "research/official_baseline_payload/modelDesign.py")
    candidate.NUM_BITS_PER_RE = 6
    candidate.PAYLOAD_BITS = 720
    transmitter = candidate.Transmitter()
    receiver = candidate.Receiver()
    assert transmitter._mod[0][0].in_features == 6
    assert receiver._fc_out.out_features == 6

    received = torch.complex(torch.randn(1, 2, 144), torch.randn(1, 2, 144))
    channel = torch.complex(torch.randn(1, 2, 16, 144), torch.randn(1, 2, 16, 144))
    output = receiver(received, channel, torch.ones(1, 5), torch.tensor([0.0]))
    assert output.shape == (1, 720)

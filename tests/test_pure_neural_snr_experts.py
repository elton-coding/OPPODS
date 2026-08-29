from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_snr_expert_boundaries() -> None:
    module = _load_module("v191_boundaries", ROOT / "research/pure_neural_v191/modelDesign.py")
    snr = torch.tensor([-20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0])
    assert module._expert_indices(snr).tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 7]


def test_replicated_experts_are_exactly_the_organizer_baseline() -> None:
    torch.manual_seed(191)
    baseline = _load_module("organizer_baseline_v191_test", ROOT / "ziliao/modelDesign.py")
    experts = _load_module("pure_neural_v191_test", ROOT / "research/pure_neural_v191/modelDesign.py")

    baseline_encoder = baseline.Encoder().eval()
    baseline_transmitter = baseline.Transmitter().eval()
    baseline_receiver = baseline.Receiver().eval()
    baseline_encoder.load_state_dict(torch.load(ROOT / "ziliao/modelSubmit/encoder.pth", weights_only=True))
    baseline_transmitter.load_state_dict(torch.load(ROOT / "ziliao/modelSubmit/transmitter.pth", weights_only=True))
    baseline_receiver.load_state_dict(torch.load(ROOT / "ziliao/modelSubmit/receiver.pth", weights_only=True))

    expert_encoder = experts.Encoder().eval()
    expert_transmitter = experts.Transmitter().eval()
    expert_receiver = experts.Receiver().eval()
    expert_encoder.initialize_from_baseline(baseline_encoder.state_dict())
    expert_transmitter.initialize_from_baseline(baseline_transmitter.state_dict())
    expert_receiver.initialize_from_baseline(baseline_receiver.state_dict())

    batch = 2
    channel = torch.complex(
        torch.randn(batch, 2, 16, 144),
        torch.randn(batch, 2, 16, 144),
    )
    user_snr = torch.tensor([-17.0, 17.0])
    baseline_feedback = torch.cat(
        [baseline_encoder(channel[index : index + 1], user_snr[index : index + 1]) for index in range(batch)]
    )
    expert_feedback = expert_encoder(channel, user_snr)
    torch.testing.assert_close(expert_feedback, baseline_feedback, rtol=0.0, atol=0.0)

    bits_list = [torch.randint(0, 2, (batch, 1152), dtype=torch.float32) for _ in range(2)]
    feedback_list = [baseline_feedback, baseline_feedback.roll(1, dims=0)]
    pair_snr = torch.tensor([[-17.0, 17.0], [-12.0, 12.0]])
    baseline_outputs = [
        baseline_transmitter(
            [bits[index : index + 1] for bits in bits_list],
            [feedback[index : index + 1] for feedback in feedback_list],
            pair_snr[:, index : index + 1],
        )
        for index in range(batch)
    ]
    baseline_signal = torch.cat([output[0] for output in baseline_outputs])
    baseline_control = torch.cat([output[1] for output in baseline_outputs])
    expert_signal, expert_control = expert_transmitter(bits_list, feedback_list, pair_snr)
    torch.testing.assert_close(expert_signal, baseline_signal, rtol=0.0, atol=0.0)
    torch.testing.assert_close(expert_control, baseline_control, rtol=0.0, atol=0.0)

    received = torch.complex(torch.randn(batch, 2, 144), torch.randn(batch, 2, 144))
    baseline_logits = torch.cat(
        [
            baseline_receiver(
                received[index : index + 1],
                channel[index : index + 1],
                baseline_control[index : index + 1],
                user_snr[index : index + 1],
            )
            for index in range(batch)
        ]
    )
    expert_logits = expert_receiver(received, channel, expert_control, user_snr)
    torch.testing.assert_close(expert_logits, baseline_logits, rtol=0.0, atol=0.0)


def test_mixed_snr_routes_gradients_to_selected_experts() -> None:
    module = _load_module("pure_neural_v191_grad_test", ROOT / "research/pure_neural_v191/modelDesign.py")
    encoder = module.Encoder()
    channel = torch.complex(torch.randn(2, 2, 16, 144), torch.randn(2, 2, 16, 144))
    feedback = encoder(channel, torch.tensor([-19.0, 19.0]))
    feedback.abs().mean().backward()
    assert any(parameter.grad is not None for parameter in encoder.experts[0].parameters())
    assert any(parameter.grad is not None for parameter in encoder.experts[7].parameters())
    assert all(parameter.grad is None for parameter in encoder.experts[3].parameters())

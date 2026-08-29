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


def test_tail_weighted_bce_emphasizes_the_worst_link() -> None:
    trainer = _load_module("pure_neural_v191_trainer_test", ROOT / "scripts/train_pure_neural_snr_experts.py")
    logits = torch.tensor([[[4.0, 4.0], [-4.0, -4.0]]])
    bits = torch.ones_like(logits)
    mean_loss = trainer.score_aligned_bce(logits, bits, tail_weight=0.0, tail_fraction=0.5)
    tail_loss = trainer.score_aligned_bce(logits, bits, tail_weight=1.0, tail_fraction=0.5)
    assert tail_loss > mean_loss


def test_asymmetric_sampling_alternates_the_band_target_user() -> None:
    trainer = _load_module("pure_neural_v195_trainer_test", ROOT / "scripts/train_pure_neural_snr_experts.py")
    generator = torch.Generator().manual_seed(195)
    snr = trainer.sample_snr(
        128,
        stage="asymmetric",
        expert_index=2,
        device=torch.device("cpu"),
        generator=generator,
    )
    rows = torch.arange(128)
    target_users = rows % 2
    target_snr = snr[rows, target_users]
    assert bool(((target_snr >= -10.0) & (target_snr < -5.0)).all())
    assert bool(((snr >= -20.0) & (snr < 20.0)).all())


def test_asymmetric_loss_downweights_the_context_user() -> None:
    trainer = _load_module("pure_neural_v196_trainer_test", ROOT / "scripts/train_pure_neural_snr_experts.py")
    logits = torch.zeros((2, 2, 4), requires_grad=True)
    bits = torch.ones_like(logits)
    loss = trainer.asymmetric_target_bce(
        logits,
        bits,
        context_weight=0.25,
        tail_weight=0.0,
        tail_fraction=0.5,
    )
    loss.backward()
    assert logits.grad is not None
    rows = torch.arange(2)
    target_gradient = logits.grad[rows, rows % 2].abs().mean()
    context_gradient = logits.grad[rows, 1 - rows % 2].abs().mean()
    torch.testing.assert_close(target_gradient, 4.0 * context_gradient)


def test_eval_mode_uses_the_registered_snr_prefix_policy() -> None:
    module = _load_module("pure_neural_v193_prefix_test", ROOT / "research/pure_neural_v191/modelDesign.py")
    receiver = module.Receiver()
    received = torch.complex(torch.randn(1, 2, 144), torch.randn(1, 2, 144))
    channel = torch.complex(torch.randn(1, 2, 16, 144), torch.randn(1, 2, 16, 144))
    control = torch.ones(1, 5)
    receiver.eval()
    assert receiver(received, channel, control, torch.tensor([-18.0])).shape == (1, 1)
    assert receiver(received, channel, control, torch.tensor([-15.0])).shape == (1, 132)
    assert receiver(received, channel, control, torch.tensor([0.0])).shape == (1, 1152)
    receiver.train()
    assert receiver(received, channel, control, torch.tensor([-18.0])).shape == (1, 1152)

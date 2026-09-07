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
    module = _load_module("v220_boundaries", ROOT / "research/pure_neural_v220/modelDesign.py")
    snr = torch.tensor([-20.0, -17.5, -15.0, 0.0, 17.49, 20.0])
    assert module._expert_indices(snr).tolist() == [0, 0, 0, 0, 0, 0]


def test_v216_components_obey_the_submission_shapes() -> None:
    torch.manual_seed(191)
    baseline = _load_module(
        "payload_baseline_v214_test",
        ROOT / "research/official_baseline_payload/modelDesign.py",
    )
    baseline.NUM_BITS_PER_RE = 7
    baseline.PAYLOAD_BITS = 1008
    experts = _load_module("pure_neural_v220_test", ROOT / "research/pure_neural_v220/modelDesign.py")

    baseline_encoder = baseline.Encoder().eval()
    baseline_transmitter = baseline.Transmitter().eval()
    baseline_receiver = baseline.Receiver().eval()

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
    assert expert_feedback.shape == baseline_feedback.shape == (batch, 96)

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
    assert expert_signal.shape == baseline_signal.shape == (batch, 16, 144)
    assert expert_control.shape == baseline_control.shape == (batch, 5)

    received = torch.complex(torch.randn(batch, 2, 144), torch.randn(batch, 2, 144))
    expert_logits = expert_receiver(received, channel, expert_control, user_snr)
    assert expert_logits.shape == (batch, 1008)
    assert torch.isfinite(expert_logits).all()


def test_mixed_snr_routes_gradients_to_selected_experts() -> None:
    module = _load_module("pure_neural_v220_grad_test", ROOT / "research/pure_neural_v220/modelDesign.py")
    encoder = module.Encoder()
    channel = torch.complex(torch.randn(2, 2, 16, 144), torch.randn(2, 2, 16, 144))
    feedback = encoder(channel, torch.tensor([-19.0, 19.0]))
    feedback.abs().mean().backward()
    assert any(parameter.grad is not None for parameter in encoder.experts[0].parameters())
    assert any(parameter.grad is not None for parameter in encoder.experts[0].parameters())
def test_tail_weighted_bce_emphasizes_the_worst_link() -> None:
    trainer = _load_module("pure_neural_v191_trainer_test", ROOT / "scripts/train_pure_neural_snr_experts.py")
    logits = torch.tensor([[[4.0, 4.0], [-4.0, -4.0]]])
    bits = torch.ones_like(logits)
    mean_loss = trainer.score_aligned_bce(logits, bits, tail_weight=0.0, tail_fraction=0.5)
    tail_loss = trainer.score_aligned_bce(logits, bits, tail_weight=1.0, tail_fraction=0.5)
    assert tail_loss > mean_loss
def test_eval_mode_uses_the_registered_snr_prefix_policy() -> None:
    module = _load_module("pure_neural_v220_prefix_test", ROOT / "research/pure_neural_v220/modelDesign.py")
    receiver = module.Receiver()
    received = torch.complex(torch.randn(1, 2, 144), torch.randn(1, 2, 144))
    channel = torch.complex(torch.randn(1, 2, 16, 144), torch.randn(1, 2, 16, 144))
    control = torch.ones(1, 5)
    receiver.eval()
    assert receiver(received, channel, control, torch.tensor([-18.0])).shape == (1, 1008)
    assert receiver(received, channel, control, torch.tensor([-15.0])).shape == (1, 1008)
    assert receiver(received, channel, control, torch.tensor([0.0])).shape == (1, 1008)
    receiver.train()
    assert receiver(received, channel, control, torch.tensor([-18.0])).shape == (1, 1008)


def test_focused_sampling_stays_in_the_requested_low_snr_range() -> None:
    trainer = _load_module("pure_neural_v221_trainer_test", ROOT / "scripts/train_pure_neural_snr_experts.py")
    generator = torch.Generator().manual_seed(221)
    snr = trainer.sample_snr(
        256,
        stage="calibrate",
        expert_index=None,
        device=torch.device("cpu"),
        generator=generator,
        focus_snr_high=-8.0,
        focus_prob=1.0,
    )
    assert torch.all(snr >= -20.0)
    assert torch.all(snr < -8.0)


def test_v222_control_encodes_the_pair_maximum_snr() -> None:
    module = _load_module("pure_neural_v222_control_test", ROOT / "research/pure_neural_v222/modelDesign.py")
    transmitter = module.Transmitter()
    batch = 2
    bits = [torch.zeros(batch, 1152) for _ in range(2)]
    feedback = [torch.zeros(batch, 96, dtype=torch.complex64) for _ in range(2)]
    _, control = transmitter(bits, feedback, torch.tensor([[-19.0, 5.0], [-10.0, 19.0]]))
    powers = torch.pow(2, torch.arange(module.NUM_CTRL))
    assert torch.sum(control.long() * powers, dim=1).tolist() == [8, 31]


def test_calibration_sampling_supports_a_high_snr_training_interval() -> None:
    trainer = _load_module("pure_neural_v223_trainer_test", ROOT / "scripts/train_pure_neural_snr_experts.py")
    generator = torch.Generator().manual_seed(223)
    snr = trainer.sample_snr(
        256,
        stage="calibrate",
        expert_index=None,
        device=torch.device("cpu"),
        generator=generator,
        train_snr_low=5.0,
        train_snr_high=20.0,
    )
    assert torch.all(snr >= 5.0)
    assert torch.all(snr < 20.0)


def test_v223_k8_uses_the_complete_1152_bit_payload() -> None:
    module = _load_module("pure_neural_v223_k8_test", ROOT / "research/pure_neural_v223_k8/modelDesign.py")
    assert module.NUM_BITS_PER_RE == 8
    assert module.PAYLOAD_BITS == 1152
    assert module.ReceiverCore()(
        torch.complex(torch.randn(1, 2, 144), torch.randn(1, 2, 144)),
        torch.complex(torch.randn(1, 2, 16, 144), torch.randn(1, 2, 16, 144)),
        torch.ones(1, 5),
        torch.tensor([10.0]),
    ).shape == (1, 1152)

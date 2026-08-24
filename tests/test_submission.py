from pathlib import Path

import torch

from modelSubmit.modelDesign import Encoder, Receiver, Transmitter


def load_champion_transmitter_weights(transmitter: Transmitter, path: Path) -> None:
    state = torch.load(path, weights_only=True)
    missing, unexpected = transmitter.load_state_dict(state, strict=False)
    assert not unexpected
    assert all(name.startswith("expert_decoders.") for name in missing)
    for expert in transmitter.expert_decoders:
        expert.load_state_dict(transmitter.decoder.state_dict())


def load_champion_receiver_weights(receiver: Receiver, path: Path) -> None:
    state = torch.load(path, weights_only=True)
    missing, unexpected = receiver.load_state_dict(state, strict=False)
    assert not unexpected
    assert all(name.startswith("llr_experts.") for name in missing)


def test_submission_components_match_official_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    encoder = Encoder()
    transmitter = Transmitter()
    receiver = Receiver()
    encoder.load_state_dict(torch.load(root / "modelSubmit/encoder.pth", weights_only=True))
    load_champion_transmitter_weights(transmitter, root / "modelSubmit/transmitter.pth")
    load_champion_receiver_weights(receiver, root / "modelSubmit/receiver.pth")
    channel = torch.complex(torch.randn(1, 2, 2, 16, 144), torch.randn(1, 2, 2, 16, 144))
    bits = [torch.randint(0, 2, (1, 1152), dtype=torch.float32) for _ in range(2)]
    snr = torch.tensor([[0.0], [20.0]])
    feedback = [encoder(channel[:, user], snr[user]) for user in range(2)]
    signal, ctrl = transmitter(bits, feedback, snr)
    received = torch.sum(channel[:, 0] * signal.unsqueeze(1), dim=2)
    llr = receiver(received, channel[:, 0], ctrl, snr[0])
    assert signal.shape == (1, 16, 144) and signal.dtype == torch.complex64
    assert ctrl.shape == (1, 5) and torch.all(ctrl == ctrl.square())
    assert llr.shape == (1, 1152) and llr.dtype == torch.float32
    assert torch.isfinite(llr).all()


def test_submission_low_snr_tail_guard_is_disabled() -> None:
    receiver = Receiver()
    channel = torch.complex(torch.randn(1, 2, 16, 144), torch.randn(1, 2, 16, 144))
    received = torch.complex(torch.randn(1, 2, 144), torch.randn(1, 2, 144))
    llr = receiver(received, channel, torch.zeros(1, 5), torch.tensor([-19.5]))
    assert llr.shape[0] == 1
    assert llr.shape[1] >= 924


def test_submission_walsh_pilot_profile_targets_weaker_user() -> None:
    root = Path(__file__).resolve().parents[1]
    encoder = Encoder()
    transmitter = Transmitter()
    receiver = Receiver()
    load_champion_transmitter_weights(transmitter, root / "modelSubmit/transmitter.pth")
    load_champion_receiver_weights(receiver, root / "modelSubmit/receiver.pth")
    channel = torch.complex(torch.randn(1, 2, 2, 16, 144), torch.randn(1, 2, 2, 16, 144))
    bits = [torch.randint(0, 2, (1, 1152), dtype=torch.float32) for _ in range(2)]
    snr = torch.tensor([[0.0], [5.0]])
    feedback = [encoder(channel[:, user], snr[user]) for user in range(2)]
    signal, ctrl = transmitter(bits, feedback, snr)
    received_weak = torch.sum(channel[:, 0] * signal.unsqueeze(1), dim=2)
    received_strong = torch.sum(channel[:, 1] * signal.unsqueeze(1), dim=2)
    weak_llr = receiver(received_weak, channel[:, 0], ctrl, snr[0])
    strong_llr = receiver(received_strong, channel[:, 1], ctrl, snr[1])
    assert torch.any(ctrl > 0)
    assert weak_llr.shape == (1, 1152)
    assert strong_llr.shape == (1, 1058)

from argparse import Namespace

import torch

from modelSubmit.modelDesign import (
    GROUPS,
    NUM_TX,
    Receiver,
    Transmitter,
    _snr_interval_value,
)
from scripts.train_precoder_aware_denoiser import validate_args


def test_interval_expert_routing_matches_shared_model_when_weights_are_equal() -> None:
    transmitter = Transmitter().eval()
    for expert in transmitter.expert_decoders:
        expert.load_state_dict(transmitter.decoder.state_dict())
    snr = torch.tensor([-20.0, -15.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, 0.0, 5.0, 10.0, 15.0])
    feedback = torch.complex(torch.randn(12, 96), torch.randn(12, 96))

    with torch.no_grad():
        expected = transmitter.decoder(feedback, snr)
        actual = transmitter._decode_feedback(feedback, snr)

    assert actual.shape == (12, GROUPS, NUM_TX)
    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_training_snr_interval_is_validated() -> None:
    validate_args(Namespace(snr_min=-10.0, snr_max=-5.0))


def test_zero_initialized_receiver_experts_preserve_llrs() -> None:
    receiver = Receiver().eval()
    llr = torch.randn(12, 1056)
    snr = torch.tensor([-20.0, -15.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, 0.0, 5.0, 10.0, 15.0])

    with torch.no_grad():
        corrected = receiver.apply_llr_expert(llr, snr)

    assert torch.equal(corrected, llr)


def test_physical_receiver_interval_profiles_use_half_open_boundaries() -> None:
    snr = torch.tensor([-2.5, -0.01, 0.0, 2.49, 2.5])

    values = _snr_interval_value(3.0, ((-2.5, 0.0, 0.3), (0.0, 2.5, 2.0)), snr)

    assert torch.equal(values, torch.tensor([0.3, 0.3, 2.0, 2.0, 3.0]))

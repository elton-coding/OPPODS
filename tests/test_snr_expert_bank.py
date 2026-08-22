from argparse import Namespace

import torch

from modelSubmit.modelDesign import GROUPS, NUM_TX, Transmitter
from scripts.train_precoder_aware_denoiser import validate_args


def test_five_db_expert_routing_matches_shared_model_when_weights_are_equal() -> None:
    transmitter = Transmitter().eval()
    for expert in transmitter.expert_decoders:
        expert.load_state_dict(transmitter.decoder.state_dict())
    snr = torch.tensor([-20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0])
    feedback = torch.complex(torch.randn(8, 96), torch.randn(8, 96))

    with torch.no_grad():
        expected = transmitter.decoder(feedback, snr)
        actual = transmitter._decode_feedback(feedback, snr)

    assert actual.shape == (8, GROUPS, NUM_TX)
    assert torch.equal(actual, expected)


def test_training_snr_interval_is_validated() -> None:
    validate_args(Namespace(snr_min=-10.0, snr_max=-5.0))

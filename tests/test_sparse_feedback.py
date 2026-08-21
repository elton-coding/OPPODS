import torch

from oppods.channel import normalize_feedback
from oppods.sparse_feedback import SparseDelayDecoder, SparseDelayEncoder, SparseDelayMUMIMOLink


def test_sparse_feedback_shapes() -> None:
    channel = torch.complex(torch.randn(3, 2, 16, 144), torch.randn(3, 2, 16, 144))
    snr = torch.tensor([-10.0, 0.0, 10.0])
    encoder = SparseDelayEncoder(mode_count=4)
    feedback = normalize_feedback(encoder(channel, snr))
    estimate = SparseDelayDecoder(mode_count=4)(feedback, snr)
    assert feedback.shape == (3, 96)
    assert estimate.shape == (3, 12, 16)
    assert torch.isfinite(estimate).all()


def test_sparse_feedback_link_shape() -> None:
    channel = torch.complex(torch.randn(1, 2, 2, 16, 144), torch.randn(1, 2, 2, 16, 144))
    bits = torch.randint(0, 2, (1, 2, 1152), dtype=torch.float32)
    snr = torch.tensor([[0.0, 5.0]])
    llr = SparseDelayMUMIMOLink(mode_count=3, decision_directed_iterations=1)(channel, bits, snr)
    assert llr.shape == (1, 2, 1152)
    assert torch.isfinite(llr).all()

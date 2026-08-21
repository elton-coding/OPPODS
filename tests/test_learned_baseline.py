import torch

from oppods.learned_baseline import LearnedFeedbackMUMIMOLink


def test_learned_feedback_link_shapes_and_finite_values() -> None:
    torch.manual_seed(4)
    link = LearnedFeedbackMUMIMOLink(width=16, layers=1, heads=2, decision_directed_iterations=1)
    channel = torch.complex(torch.randn(2, 2, 2, 16, 144), torch.randn(2, 2, 2, 16, 144))
    bits = torch.randint(0, 2, (2, 2, 1152), dtype=torch.float32)
    snr = torch.tensor([[0.0, 5.0], [-5.0, 10.0]])
    llr = link(channel, bits, snr)
    assert llr.shape == (2, 2, 1152)
    assert llr.dtype == torch.float32
    assert torch.isfinite(llr).all()

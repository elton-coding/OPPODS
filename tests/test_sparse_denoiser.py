import torch

from oppods.sparse_denoiser import SparseFeedbackDenoiser


def test_sparse_denoiser_starts_at_wiener_and_backpropagates() -> None:
    denoiser = SparseFeedbackDenoiser(width=16, layers=1, heads=2)
    feedback = torch.complex(torch.randn(4, 96), torch.randn(4, 96))
    snr = torch.tensor([-10.0, 0.0, 10.0, 20.0])
    result = denoiser.forward_details(feedback, snr)
    assert result.effective_channel.shape == (4, 12, 16)
    assert torch.equal(result.coefficients, result.wiener_coefficients)
    result.effective_channel.abs().mean().backward()
    assert denoiser.output.weight.grad is not None
    assert torch.isfinite(denoiser.output.weight.grad).all()

import torch

from oppods.analytic_baseline import _task_feedback
from oppods.channel import normalize_feedback
from oppods.feedback_model import TaskFeedbackDecoder, TaskOrientedEncoder, feedback_reconstruction_loss


def test_feedback_model_shapes_and_backward() -> None:
    generator = torch.Generator().manual_seed(1176)
    channel = torch.complex(
        torch.randn((4, 2, 16, 144), generator=generator),
        torch.randn((4, 2, 16, 144), generator=generator),
    )
    snr = torch.linspace(-20.0, 20.0, 4)
    encoder = TaskOrientedEncoder(width=32, layers=1, heads=4)
    decoder = TaskFeedbackDecoder(width=32, layers=1, heads=4)
    feedback = normalize_feedback(encoder(channel, snr))
    prediction = decoder(feedback, snr)
    target = _task_feedback(channel, group_size=12)
    loss, metrics = feedback_reconstruction_loss(prediction, target, snr)
    loss.backward()

    assert feedback.shape == (4, 96)
    assert prediction.effective_channel.shape == (4, 12, 16)
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["beam_alignment"])
    assert any(parameter.grad is not None for parameter in encoder.parameters())
    assert any(parameter.grad is not None for parameter in decoder.parameters())

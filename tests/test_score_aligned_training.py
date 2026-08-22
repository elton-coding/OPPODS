import torch

from scripts.train_score_aligned_denoiser import anchor_penalty


def test_anchor_penalty_is_zero_at_initial_checkpoint() -> None:
    parameters = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([-3.0])}
    anchors = {name: value.clone() for name, value in parameters.items()}

    assert anchor_penalty(parameters, anchors).item() == 0.0


def test_anchor_penalty_averages_parameter_tensors() -> None:
    parameters = {"a": torch.tensor([1.0, 3.0]), "b": torch.tensor([2.0])}
    anchors = {"a": torch.zeros(2), "b": torch.zeros(1)}

    assert torch.isclose(anchor_penalty(parameters, anchors), torch.tensor(4.5))

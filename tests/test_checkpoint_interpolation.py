import pytest
import torch

from scripts.interpolate_denoiser_checkpoints import interpolate_state_dicts


def test_interpolation_uses_candidate_alpha() -> None:
    base = {"weight": torch.tensor([0.0, 2.0]), "count": torch.tensor(3)}
    candidate = {"weight": torch.tensor([4.0, 6.0]), "count": torch.tensor(3)}

    mixed = interpolate_state_dicts(base, candidate, 0.25)

    assert torch.equal(mixed["weight"], torch.tensor([1.0, 3.0]))
    assert mixed["count"].item() == 3


def test_interpolation_rejects_incompatible_state() -> None:
    with pytest.raises(ValueError, match="different keys"):
        interpolate_state_dicts({"a": torch.ones(1)}, {"b": torch.ones(1)}, 0.5)

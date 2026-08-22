from argparse import Namespace

import pytest
import torch

from scripts.train_precoder_aware_denoiser import sample_training_snr, validate_args


def training_args(**overrides: float | str | None) -> Namespace:
    values: dict[str, float | str | None] = {
        "tail_alpha": 0.1,
        "tail_weight": 0.2,
        "tail_target": "weakest-user",
        "weakest_user_weight": 0.2,
        "weak_user_focus_probability": 1.0,
        "weak_user_focus_min_snr": -15.5,
        "weak_user_focus_max_snr": -9.5,
        "minimum_profile_max_snr": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_weak_user_focus_places_one_user_in_target_interval() -> None:
    args = training_args()
    validate_args(args)
    generator = torch.Generator().manual_seed(1176)
    snr = sample_training_snr(128, torch.device("cpu"), generator, args)

    focused = (snr >= args.weak_user_focus_min_snr) & (snr < args.weak_user_focus_max_snr)
    assert torch.all(focused.any(dim=1))
    assert torch.all((-20.0 <= snr) & (snr <= 20.0))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"tail_alpha": 0.0}, "tail-alpha"),
        ({"tail_weight": 0.8, "weakest_user_weight": 0.3}, "must not exceed"),
        ({"weak_user_focus_probability": 1.1}, "probability"),
        ({"weak_user_focus_min_snr": -9.5, "weak_user_focus_max_snr": -15.5}, "minimum SNR"),
    ],
)
def test_invalid_fairness_arguments_are_rejected(
    overrides: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_args(training_args(**overrides))

import torch

from modelSubmit.modelDesign import (
    PILOT_STEERING_SHRINKAGE,
    PILOT_STEERING_SHRINKAGE_INTERVALS,
    _snr_interval_value,
    _snr_pair_interval_value,
)


def test_physical_receiver_interval_profiles_use_half_open_boundaries() -> None:
    snr = torch.tensor([-2.5, -0.01, 0.0, 2.49, 2.5])

    values = _snr_interval_value(3.0, ((-2.5, 0.0, 0.3), (0.0, 2.5, 2.0)), snr)

    assert torch.equal(values, torch.tensor([0.3, 0.3, 2.0, 2.0, 3.0]))


def test_shared_link_profile_routes_when_any_user_is_in_interval() -> None:
    snr_by_user = torch.tensor(
        [
            [-10.0, 12.0],
            [-2.75, -10.1],
            [-2.8, 19.0],
            [0.0, 5.0],
            [-1.0, 5.0],
            [4.0, 5.0],
        ]
    )

    values = _snr_pair_interval_value(
        0.45,
        ((-10.0, -2.75, 0.4),),
        snr_by_user,
        ((-0.75, 20.0, 0.9), (4.0, 20.0, 1.8)),
    )

    assert torch.allclose(values, torch.tensor([0.4, 0.45, 0.4, 0.9, 0.45, 1.8]))


def test_pilot_steering_profile_uses_half_open_snr_intervals() -> None:
    snr = torch.tensor([1.9999, 2.0, 7.4999, 7.5])

    values = _snr_interval_value(
        PILOT_STEERING_SHRINKAGE,
        PILOT_STEERING_SHRINKAGE_INTERVALS,
        snr,
    )

    assert torch.allclose(values, torch.tensor([0.0375, 0.0125, 0.0125, 0.0375]))

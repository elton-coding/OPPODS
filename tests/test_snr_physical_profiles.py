import torch

from modelSubmit.modelDesign import (
    DATA_GAIN_SOFT_TEMPERATURE,
    DATA_GAIN_SOFT_TEMPERATURE_INTERVALS,
    INTERFERENCE_CANCELLATION_SCALE,
    INTERFERENCE_CANCELLATION_SCALE_INTERVALS,
    _middle_extension_confidence,
    _snr_interval_value,
    _snr_pair_interval_value,
)


def test_middle_extension_q25_over_mean_confidence_is_computed_per_user() -> None:
    extension_llr = torch.tensor(
        [
            [0.0, -1.0, 2.0, -3.0, 4.0],
            [0.0, -2.0, 4.0, -6.0, 8.0],
        ]
    )

    confidence = _middle_extension_confidence(extension_llr)

    assert torch.allclose(confidence, torch.tensor([0.5, 0.5]))


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


def test_interference_cancellation_scale_profile_uses_half_open_snr_intervals() -> None:
    snr = torch.tensor([-7.7501, -7.75, -5.0001, -5.0, -0.25, 2.7499, 2.75, 8.0, 17.9999, 18.0])

    values = _snr_interval_value(
        INTERFERENCE_CANCELLATION_SCALE,
        INTERFERENCE_CANCELLATION_SCALE_INTERVALS,
        snr,
    )

    assert torch.allclose(
        values,
        torch.tensor([0.2, 0.25, 0.25, 0.2, 0.25, 0.25, 0.2, 0.25, 0.25, 0.2]),
    )


def test_data_gain_temperature_profile_uses_half_open_snr_intervals() -> None:
    snr = torch.tensor([10.4999, 10.5, 15.7499, 15.75])

    values = _snr_interval_value(
        DATA_GAIN_SOFT_TEMPERATURE,
        DATA_GAIN_SOFT_TEMPERATURE_INTERVALS,
        snr,
    )

    assert torch.allclose(values, torch.tensor([0.5, 0.6, 0.6, 0.5]))

import torch

from scripts.average_receiver_checkpoints import average_state_dicts


def test_average_state_dicts() -> None:
    states = [
        {"weight": torch.tensor([1.0, 3.0]), "count": torch.tensor(2)},
        {"weight": torch.tensor([3.0, 5.0]), "count": torch.tensor(2)},
    ]

    averaged = average_state_dicts(states)

    assert torch.equal(averaged["weight"], torch.tensor([2.0, 4.0]))
    assert averaged["count"].item() == 2

import torch

from oppods.reserved_pilot_link import ReservedPilotMUMIMOLink


def test_reserved_pilot_link_shapes() -> None:
    channel = torch.complex(torch.randn(2, 2, 2, 16, 144), torch.randn(2, 2, 2, 16, 144))
    bits = torch.randint(0, 2, (2, 2, 1152), dtype=torch.float32)
    snr = torch.tensor([[0.0, 5.0], [10.0, -5.0]])
    llr, identity = ReservedPilotMUMIMOLink()(channel, bits, snr)
    assert llr.shape == (2, 2, 960)
    assert identity.shape == (2, 2)
    assert torch.isfinite(llr).all()

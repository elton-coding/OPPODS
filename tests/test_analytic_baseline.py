import torch

from oppods.analytic_baseline import AnalyticMUMIMOLink


def test_analytic_link_shapes_and_finiteness() -> None:
    generator = torch.Generator().manual_seed(1176)
    channel = torch.complex(
        torch.randn((3, 2, 2, 16, 144), generator=generator),
        torch.randn((3, 2, 2, 16, 144), generator=generator),
    )
    bits = torch.randint(0, 2, (3, 2, 1152), generator=generator, dtype=torch.float32)
    snr = torch.tensor([[-10.0, 0.0], [5.0, 10.0], [15.0, 20.0]])
    link = AnalyticMUMIMOLink()
    llr = link(channel, bits, snr, generator=generator)
    assert llr.shape == (3, 2, 1152)
    assert llr.dtype == torch.float32
    assert torch.isfinite(llr).all()


def test_analytic_link_can_return_progressive_prefix() -> None:
    generator = torch.Generator().manual_seed(2176)
    channel = torch.complex(
        torch.randn((1, 2, 2, 16, 144), generator=generator),
        torch.randn((1, 2, 2, 16, 144), generator=generator),
    )
    bits = torch.randint(0, 2, (1, 2, 1152), generator=generator, dtype=torch.float32)
    snr = torch.zeros((1, 2))
    link = AnalyticMUMIMOLink(bits_per_symbol=8, decoded_layers=2)
    llr = link(channel, bits, snr, generator=generator)
    assert llr.shape == (1, 2, 288)

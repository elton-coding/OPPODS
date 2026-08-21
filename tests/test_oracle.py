import torch

from oppods.oracle import _dominant_hermitian_eigenvector_2x2, dominant_receive_combiner, rzf_precoder


def test_closed_form_dominant_eigenvector() -> None:
    generator = torch.Generator().manual_seed(1176)
    channel = torch.complex(
        torch.randn((64, 2, 5), generator=generator),
        torch.randn((64, 2, 5), generator=generator),
    )
    covariance = channel @ channel.conj().transpose(-2, -1)
    vector = _dominant_hermitian_eigenvector_2x2(covariance)
    rayleigh = torch.einsum("bi,bij,bj->b", vector.conj(), covariance, vector).real
    largest = torch.linalg.eigvalsh(covariance)[..., -1]
    assert torch.allclose(torch.linalg.vector_norm(vector, dim=-1), torch.ones(64), atol=1e-5)
    assert torch.allclose(rayleigh, largest, atol=2e-5, rtol=2e-5)


def test_group_size_one_large_batch_is_supported() -> None:
    generator = torch.Generator().manual_seed(1177)
    channel = torch.complex(
        torch.randn((128, 2, 2, 16, 144), generator=generator),
        torch.randn((128, 2, 2, 16, 144), generator=generator),
    )
    combiner = dominant_receive_combiner(channel, group_size=1)
    assert combiner.shape == (128, 2, 144, 2)
    assert torch.isfinite(combiner).all()


def test_fairness_power_allocation_is_normalized() -> None:
    generator = torch.Generator().manual_seed(1178)
    effective = torch.complex(
        torch.randn((7, 12, 2, 16), generator=generator),
        torch.randn((7, 12, 2, 16), generator=generator),
    )
    noise = torch.rand((7, 2), generator=generator) + 0.1
    precoder = rzf_precoder(effective, noise, fairness_exponent=0.5)
    energy = torch.sum(torch.abs(precoder).square(), dim=(-2, -1))
    assert torch.allclose(energy, torch.ones_like(energy), atol=1e-5)

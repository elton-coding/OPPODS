import pytest
import torch

from oppods.modulation import (
    SUPPORTED_BITS_PER_SYMBOL,
    layered_qam_maxlog_llr,
    layered_qam_modulate,
    qam_maxlog_llr,
    qam_modulate,
)


@pytest.mark.parametrize("bits_per_symbol", SUPPORTED_BITS_PER_SYMBOL)
def test_noiseless_roundtrip(bits_per_symbol: int) -> None:
    generator = torch.Generator().manual_seed(1176 + bits_per_symbol)
    bits = torch.randint(0, 2, (8, 12 * bits_per_symbol), generator=generator, dtype=torch.float32)
    symbols = qam_modulate(bits, bits_per_symbol)
    llr = qam_maxlog_llr(symbols, torch.ones_like(symbols), torch.full_like(symbols.real, 1e-3), bits_per_symbol)
    assert torch.equal(llr >= 0, bits >= 0.5)


@pytest.mark.parametrize("bits_per_symbol", SUPPORTED_BITS_PER_SYMBOL)
def test_noiseless_layered_roundtrip(bits_per_symbol: int) -> None:
    generator = torch.Generator().manual_seed(2176 + bits_per_symbol)
    bits = torch.randint(0, 2, (8, 12 * bits_per_symbol), generator=generator, dtype=torch.float32)
    symbols = layered_qam_modulate(bits, bits_per_symbol)
    llr = layered_qam_maxlog_llr(
        symbols,
        torch.ones_like(symbols),
        torch.full_like(symbols.real, 1e-3),
        bits_per_symbol,
    )
    assert torch.equal(llr >= 0, bits >= 0.5)


def test_nonuniform_qam_noiseless_roundtrip() -> None:
    bits = torch.randint(0, 2, (8, 96), dtype=torch.float32)
    symbols = layered_qam_modulate(bits, 8, central_boost=2.0)
    llr = layered_qam_maxlog_llr(
        symbols,
        torch.ones_like(symbols),
        torch.full_like(symbols.real, 1e-3),
        8,
        central_boost=2.0,
    )
    assert torch.equal(llr >= 0, bits >= 0.5)

import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_k4_payload_uses_all_144_resources_and_ignores_unsent_bits():
    module = load(ROOT / "research/pure_neural_v236/modelDesign.py")
    bits = torch.arange(1152).float()[None].repeat(2, 1)
    snr = torch.tensor([-18., 10.])
    packed = module._pack_transmit_bits(bits, snr)
    assert packed[0].shape == (144, 8)
    assert torch.equal(packed[0, :, :4], bits[0, :576].reshape(144, 4))
    assert torch.equal(packed[0, :, 4:], torch.full((144, 4), .5))
    assert torch.equal(packed[1], bits[1].reshape(144, 8))
    altered = bits.clone()
    altered[0, 576:] = -100
    assert torch.equal(module._pack_transmit_bits(altered, snr), packed)
    unpacked = module._pack_receive_logits(bits, snr)
    assert torch.equal(unpacked[0, :576], bits[0].reshape(144, 8)[:, :4].flatten())
    assert torch.count_nonzero(unpacked[0, 576:]) == 0
    assert torch.equal(unpacked[1], bits[1])
    assert module.payload_lengths(torch.tensor([-20., -15., 20.])).tolist() == [576, 1152, 1152]


def test_untransmitted_bits_receive_exact_neutral_credit_and_no_gradient():
    trainer = load(ROOT / "scripts/train_pure_neural_snr_experts.py")
    logits = torch.ones(1, 2, 1152, requires_grad=True)
    bits = torch.ones_like(logits)
    lengths = torch.tensor([[576, 1152]])
    scores = trainer.official_scores_from_logits(logits, bits, lengths)
    torch.testing.assert_close(scores, torch.tensor([[75., 100.]]), atol=0, rtol=0)
    changed = logits.detach().clone()
    changed[0, 0, 576:] = -100
    assert torch.equal(scores, trainer.official_scores_from_logits(changed, bits, lengths))
    for kind in ("soft_score", "hard_rank_score"):
        kwargs = {"loss_kind": kind, "margin": .5, "tail_weight": 0., "tail_fraction": .1,
                  "valid_lengths": lengths}
        actual = trainer.score_aligned_loss(logits, bits, **kwargs)
        expected = trainer.score_aligned_loss(changed, bits, **kwargs)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        grad = torch.autograd.grad(actual, logits)[0]
        assert torch.count_nonzero(grad[0, 0, 576:]) == 0
        assert grad[0, 0, :576].abs().sum() > 0
    with pytest.raises(ValueError, match="positive"):
        trainer.official_scores_from_logits(logits, bits, torch.tensor([[0, 1152]]))


def test_high_snr_parent_parity_and_mixed_single_sample_padding():
    torch.set_num_threads(2)
    torch.manual_seed(236)
    trainer = load(ROOT / "scripts/train_pure_neural_snr_experts.py")
    parent = trainer.PureNeuralLink(load(ROOT / "research/pure_neural_v227/modelDesign.py")).eval()
    candidate = trainer.PureNeuralLink(load(ROOT / "research/pure_neural_v236/modelDesign.py")).eval()
    candidate.load_state_dict(parent.state_dict(), strict=True)
    assert sum(p.numel() for p in candidate.parameters()) == 48191288
    h = torch.randn(2, 2, 2, 16, 144, dtype=torch.complex64)
    bits = torch.randint(0, 2, (2, 2, 1152)).float()
    snr = torch.tensor([[-12., 10.], [5., 15.]])
    with torch.no_grad():
        expected = parent(h, bits, snr, generator=torch.Generator().manual_seed(236))
        actual = candidate(h, bits, snr, generator=torch.Generator().manual_seed(236))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        mixed = candidate(h[:1], bits[:1], torch.tensor([[-18., 10.]]),
                          generator=torch.Generator().manual_seed(236))
    assert mixed.shape == (1, 2, 1152)
    assert torch.count_nonzero(mixed[0, 0, 576:]) == 0
    # Official Receiver alone must return the real payload, never padded guesses.
    y = torch.randn(1, 2, 144, dtype=torch.complex64)
    control = torch.zeros(1, 5)
    with torch.no_grad():
        assert candidate.receiver(y, h[:1, 0], control, torch.tensor([-18.])).shape == (1, 576)
        assert candidate.receiver(y, h[:1, 0], control, torch.tensor([10.])).shape == (1, 1152)


def test_fixed_payload_scoring_matches_existing_formula_exactly():
    trainer = load(ROOT / "scripts/train_pure_neural_snr_experts.py")
    torch.manual_seed(236)
    for width in (1008, 1152):
        logits = torch.randn(7, 2, width)
        bits = torch.randint(0, 2, (7, 2, 1152)).float()
        correct = ((logits >= 0) == (bits[..., :width] >= .5)).sum(-1)
        expected = 100. * (correct + .5 * (1152 - width)) / 1152
        assert torch.equal(trainer.official_scores_from_logits(logits, bits), expected)

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


def test_eight_expert_refinement_preserves_the_pretrained_link(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(230)
    trainer = load(ROOT / "scripts/train_pure_neural_snr_experts.py")
    old_design = load(ROOT / "research/pure_neural_v227/modelDesign.py")
    new_design = load(ROOT / "research/pure_neural_v230/modelDesign.py")
    old, new = trainer.PureNeuralLink(old_design), trainer.PureNeuralLink(new_design)
    old.save_submission(tmp_path, ROOT / "research/pure_neural_v227/modelDesign.py")
    new.initialize_from_expert_bank(tmp_path, [0, 0, 1, 1, 1, 1, 1, 1])
    assert sum(p.numel() for p in new.parameters()) == 190354976
    assert new_design._expert_indices(torch.arange(-20., 21., 5.)).tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 7]
    torch.manual_seed(230)
    h = torch.randn(8, 2, 2, 16, 144, dtype=torch.complex64)
    bits = torch.randint(0, 2, (8, 2, 1152)).float()
    snr = torch.stack([torch.arange(-19., 20., 5.), torch.full((8,), 19.)], dim=1)
    with torch.no_grad():
        expected = old(h, bits, snr, generator=torch.Generator().manual_seed(230))
        actual = new(h, bits, snr, generator=torch.Generator().manual_seed(230))
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=1e-4)
    assert torch.equal(actual >= 0, expected >= 0)
    with pytest.raises(ValueError, match="needs --baseline-expert-map"):
        new.initialize_from_baseline(tmp_path)

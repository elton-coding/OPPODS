import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_encoder_unfreeze_restores_end_to_end_gradients():
    torch.set_num_threads(2)
    torch.manual_seed(235)
    trainer = load(ROOT / "scripts/train_pure_neural_snr_experts.py")
    design = load(ROOT / "research/pure_neural_v227/modelDesign.py")
    link = trainer.PureNeuralLink(design)
    frozen = trainer.select_trainable_parameters(link, ["transmitter", "receiver"], None)
    assert sum(p.numel() for p in frozen) == 47387896
    assert all(not p.requires_grad for p in link.encoder.parameters())
    full = trainer.select_trainable_parameters(link, ["encoder", "transmitter", "receiver"], None)
    assert sum(p.numel() for p in full) == 48191288
    assert len({id(p) for p in full}) == len(full)
    assert all(p.requires_grad for p in link.parameters())
    channel = torch.randn(2, 2, 2, 16, 144, dtype=torch.complex64)
    bits = torch.randint(0, 2, (2, 2, 1152)).float()
    snr = torch.tensor([[-15., 10.], [5., 15.]])
    logits = link(channel, bits, snr, generator=torch.Generator().manual_seed(235))
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, bits)
    loss.backward()
    for component in (link.encoder, link.transmitter, link.receiver):
        assert sum(float(p.grad.abs().sum()) for p in component.parameters() if p.grad is not None) > 0

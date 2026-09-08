from pathlib import Path
import importlib.util

import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cloned_joint_profiles_preserve_outputs_and_freeze_other_weights():
    torch.set_num_threads(2)
    torch.manual_seed(227)
    trainer = load(ROOT / "scripts/train_pure_neural_snr_experts.py")
    design = load(ROOT / "research/pure_neural_v227/modelDesign.py")
    baseline = load(ROOT / "research/pure_neural_v223_k8/modelDesign.py")
    old = trainer.PureNeuralLink(baseline)
    new = trainer.PureNeuralLink(design)
    for name in ("encoder", "transmitter", "receiver"):
        getattr(new, name).initialize_from_baseline(getattr(old, name).experts[0].state_dict())
    assert len(new.encoder.experts) == 1
    assert design._expert_indices(torch.tensor([-20., -10.01, -10., 20.])).tolist() == [0, 0, 1, 1]
    channel = torch.randn(4, 2, 2, 16, 144, dtype=torch.complex64)
    bits = torch.randint(0, 2, (4, 2, 1152)).float()
    snr = torch.tensor([[-15., 12.], [12., -15.], [-5., 5.], [-10., -10.]])
    with torch.no_grad():
        expected = old(channel, bits, snr, generator=torch.Generator().manual_seed(50))
        actual = new(channel, bits, snr, generator=torch.Generator().manual_seed(50))
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    parameters = trainer.select_trainable_parameters(new, ["transmitter", "receiver"], 0)
    frozen = {name: p.detach().clone() for name, p in new.named_parameters() if not p.requires_grad}
    assert frozen and parameters
    optimizer = torch.optim.Adam(parameters, lr=1e-4)
    prediction = new(channel, bits, snr, generator=torch.Generator().manual_seed(50))
    torch.nn.functional.binary_cross_entropy_with_logits(prediction, bits).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in parameters)
    optimizer.step()
    for name, parameter in new.named_parameters():
        if name in frozen:
            assert parameter.grad is None
            assert torch.equal(parameter, frozen[name])
    with pytest.raises(ValueError, match="encoder has no expert"):
        trainer.select_trainable_parameters(new, ["encoder"], 1)

import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_control_preserves_profile_and_bounds_partner_error():
    module = load(ROOT / "research/pure_neural_v233/modelDesign.py")
    torch.manual_seed(233)
    snr = torch.rand(2, 1000) * 40 - 20
    profile = module._expert_indices(snr.min(dim=0).values)
    control = module._control_bits(snr, profile)
    assert control.shape == (1000, 5)
    assert torch.equal(control, control.square())
    assert torch.equal(control[:, 0].long(), profile)
    for own in (0, 1):
        error = module._partner_snr(control, snr[own]) - snr[1 - own]
        assert error.abs().max() <= 2.50001


def test_zero_partner_embedding_preserves_parent_receiver():
    torch.set_num_threads(2)
    old = load(ROOT / "research/pure_neural_v227/modelDesign.py").ReceiverCore()
    module = load(ROOT / "research/pure_neural_v233/modelDesign.py")
    new = module.ReceiverCore()
    new.load_parent_state(old.state_dict())
    y, h = torch.randn(2, 2, 144, dtype=torch.complex64), torch.randn(2, 2, 16, 144, dtype=torch.complex64)
    snr = torch.tensor([[-18., 10.], [15., 16.]])
    control = module._control_bits(snr, module._expert_indices(snr.min(0).values))
    actual = new(y, h, control, snr[0])
    torch.testing.assert_close(actual, old(y, h, control, snr[0]), atol=0, rtol=0)
    actual.square().mean().backward()
    assert new._partner_embed.weight.grad.abs().sum() > 0


def test_receiver_does_not_treat_context_bits_as_expert_id():
    module = load(ROOT / "research/pure_neural_v233/modelDesign.py")

    class Constant(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = value

        def forward(self, y, h, control, snr):
            return torch.full((y.shape[0], 1152), self.value, dtype=torch.float32)

    receiver = module.Receiver()
    receiver.experts = torch.nn.ModuleList([Constant(0.), Constant(1.)])
    y, h = torch.zeros(2, 2, 144, dtype=torch.complex64), torch.zeros(2, 2, 16, 144, dtype=torch.complex64)
    output = receiver(y, h, torch.tensor([[0., 1., 1., 1., 1.], [1., 0., 0., 0., 0.]]), torch.zeros(2))
    assert torch.equal(output[:, 0], torch.tensor([0., 1.]))

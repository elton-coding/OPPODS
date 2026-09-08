import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load(version):
    spec = importlib.util.spec_from_file_location(version, ROOT / f"research/pure_neural_{version}/modelDesign.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("component", ["TransmitterCore", "ReceiverCore"])
def test_width_morphism_preserves_function_and_breaks_gradient_symmetry(component):
    torch.set_num_threads(2)
    torch.manual_seed(238)
    old = getattr(load("v227"), component)().eval()
    new = getattr(load("v238"), component)().eval()
    new.load_parent_state(old.state_dict())
    if component == "TransmitterCore":
        args = ([torch.randint(0, 2, (1, 1152)).float() for _ in range(2)],
                [torch.randn(1, 96, dtype=torch.complex64) for _ in range(2)],
                torch.tensor([[-18.], [12.]]))
        expected = old(*args)[0]
        actual = new(*args)[0]
        loss = actual.abs().square().mean()
    else:
        args = (torch.randn(1, 2, 144, dtype=torch.complex64),
                torch.randn(1, 2, 16, 144, dtype=torch.complex64),
                torch.zeros(1, 5), torch.tensor([-12.]))
        expected = old(*args)
        actual = new(*args)
        loss = actual.square().mean()
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    loss.backward()
    gradient = new._embed.weight.grad
    assert (gradient[:512] - gradient[512:]).abs().max() > 1e-9
    for name, value in old.state_dict().items():
        if name.startswith(("_bit_embed.", "_feedback_expand.")):
            torch.testing.assert_close(new.state_dict()[name], value, atol=0, rtol=0)
    broken = dict(old.state_dict())
    del broken["_embed.weight"]
    with pytest.raises(ValueError, match="invalid parent"):
        new.load_parent_state(broken)


def test_checkpoint_recomputation_preserves_forward_and_gradients():
    torch.set_num_threads(2)
    torch.manual_seed(238)
    design = load("v238")
    blocks = torch.nn.ModuleList([design.ResidualMLPBlock(32) for _ in range(2)])
    values = torch.randn(2, 3, 32, requires_grad=True)
    design.CHECKPOINT_TRAINING = False
    plain = design._run_blocks(blocks, values, True)
    plain.square().sum().backward()
    gradients = [p.grad.clone() for p in blocks.parameters()]
    input_gradient = values.grad.clone()
    blocks.zero_grad(set_to_none=True)
    values.grad = None
    design.CHECKPOINT_TRAINING = True
    recomputed = design._run_blocks(blocks, values, True)
    recomputed.square().sum().backward()
    torch.testing.assert_close(plain, recomputed, atol=0, rtol=0)
    torch.testing.assert_close(values.grad, input_gradient, atol=0, rtol=0)
    for expected, parameter in zip(gradients, blocks.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, expected, atol=0, rtol=0)

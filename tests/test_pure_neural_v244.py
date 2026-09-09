import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load(version):
    spec = importlib.util.spec_from_file_location(version, ROOT / f"research/pure_neural_{version}/modelDesign.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("kind", ["TransmitterCore", "ReceiverCore"])
def test_token_context_exact_parent_function_and_first_step_gradient(kind):
    torch.set_num_threads(2)
    torch.manual_seed(244)
    parent = getattr(load("v227"), kind)()
    candidate = getattr(load("v244"), kind)()
    candidate.load_parent_state(parent.state_dict())
    if kind == "TransmitterCore":
        args = ([torch.randint(0, 2, (2, 1152)).float() for _ in range(2)],
                [torch.randn(2, 96, dtype=torch.complex64) for _ in range(2)],
                torch.tensor([[-18., 10.], [15., 16.]]))
        actual, control = candidate(*args)
        expected, expected_control = parent(*args)
        torch.testing.assert_close(control, expected_control, atol=0, rtol=0)
    else:
        args = (torch.randn(2, 2, 144, dtype=torch.complex64),
                torch.randn(2, 2, 16, 144, dtype=torch.complex64),
                torch.zeros(2, 5), torch.tensor([-18., 10.]))
        actual, expected = candidate(*args), parent(*args)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.abs().square().mean().backward()
    gradient = candidate._token_context.up.weight.grad
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    broken = dict(parent.state_dict())
    del broken["_embed.weight"]
    with pytest.raises(ValueError, match="invalid parent"):
        candidate.load_parent_state(broken)


def test_token_context_can_transport_information_between_subcarriers():
    torch.manual_seed(245)
    layer = load("v244").TokenContext(width=8, bottleneck=4)
    values = torch.randn(1, 144, 8)
    torch.testing.assert_close(layer(values), values, atol=0, rtol=0)
    with torch.no_grad():
        layer.up.weight.normal_(0, .1)
        layer.mix.weight.zero_()
        layer.mix.weight[100, 0] = 1
    modified = values.clone()
    modified[0, 0, 0] += 2
    delta = layer(modified) - layer(values)
    assert delta[0, 100].abs().sum() > 0
    assert torch.count_nonzero(delta[0, 1:100]) == 0


def test_context_cpu_probe_has_no_cuda_cap_and_gpu_uses_full_training_batch():
    from run_pure_neural_context_v244 import PLAN, probe_command, train_command

    cpu, gpu = probe_command("cpu"), probe_command("cuda")
    assert "--gpu-memory-fraction" not in cpu
    assert gpu[gpu.index("--gpu-memory-fraction") + 1] == "0.4"
    assert gpu[gpu.index("--batch-size") + 1] == "100"
    assert cpu[cpu.index("--batch-size") + 1] == "2"
    formal = train_command(PLAN)
    assert formal[formal.index("--steps") + 1] == "12000"
    assert formal[formal.index("--loss-kind") + 1] == "soft_score"

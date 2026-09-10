import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import probe_pure_neural_shared_endtoend_v262 as probe


def toy_independent():
    link = nn.Module()
    link.encoder = nn.Linear(1, 1, bias=False)
    for component in ("transmitter", "receiver"):
        bank = nn.Module()
        bank.experts = nn.ModuleList()
        for index in range(8):
            expert = nn.Module()
            expert._embed = nn.Linear(1, 1, bias=False)
            expert._out = nn.Linear(1, 1, bias=False)
            for parameter in expert.parameters():
                parameter.grad = torch.full_like(parameter, index + 1.)
            bank.experts.append(expert)
        setattr(link, component, bank)
    link.encoder.weight.grad = torch.full_like(link.encoder.weight, 9.)
    return link


def test_grouped_gradients_sum_shared_keep_private_and_encoder():
    link = toy_independent()
    before = {name: p.grad.clone() for name, p in link.named_parameters()}
    result = probe.grouped_independent_gradients(link)
    assert result["encoder.weight"].item() == 9
    for component in ("transmitter", "receiver"):
        assert result[f"{component}.experts.0._embed.weight"].item() == 3
        assert result[f"{component}.experts.2._embed.weight"].item() == 33
        for index in range(8):
            assert result[f"{component}.experts.{index}._out.weight"].item() == index+1
    assert len(result) == 21
    assert all(torch.equal(p.grad, before[name]) for name, p in link.named_parameters())


@pytest.mark.parametrize("invalid", [None, float("nan")])
def test_missing_or_nonfinite_independent_gradient_rejected(invalid):
    link = toy_independent()
    link.encoder.weight.grad = None if invalid is None else torch.full_like(link.encoder.weight, invalid)
    with pytest.raises(ValueError):
        probe.grouped_independent_gradients(link)


@pytest.fixture(scope="module")
def real_link():
    torch.set_num_threads(2)
    module = probe.load_model_design(probe.ROOT / probe.DESIGN)
    return probe.PureNeuralLink(module)


def test_actual_shared_architecture_keeps_encoder_trainability(real_link):
    parameters = probe.select_trainable_parameters(real_link, probe.COMPONENTS)
    assert len(parameters) == 532
    assert sum(p.numel() for p in parameters) == 73547840
    assert probe.assert_shared_aliases(real_link) == 1024
    assert all(p.requires_grad for p in real_link.parameters())


def test_private_suffix_must_not_be_tied(real_link):
    original = real_link.receiver.experts[1]._blocks[8]
    try:
        real_link.receiver.experts[1]._blocks[8] = real_link.receiver.experts[0]._blocks[8]
        with pytest.raises(AssertionError):
            probe.assert_shared_aliases(real_link)
    finally:
        real_link.receiver.experts[1]._blocks[8] = original


def test_scope_fixed_full_payload_and_same_parent_mapping():
    assert probe.DESIGN == "research/pure_neural_v257/modelDesign.py"
    assert probe.CONTROL_DESIGN == "research/pure_neural_v230/modelDesign.py"
    assert probe.MAPPING == [0, 0, 1, 1, 1, 1, 1, 1]
    assert probe.COMPONENTS == ["encoder", "transmitter", "receiver"]
    assert probe.CPU_PROOF != "benchmarks/v258_cpu_endtoend_probe.json"

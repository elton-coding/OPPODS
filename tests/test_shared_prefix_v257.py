from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from probe_pure_neural_shared_v257 import (
    DESIGN,
    MAPPING,
    assert_aliases,
    canonical_name,
    shared_name,
)
from train_pure_neural_snr_experts import PureNeuralLink, load_model_design

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def link():
    torch.manual_seed(257)
    return PureNeuralLink(load_model_design(ROOT / DESIGN))


def test_exact_parent_group_and_eight_routes():
    module = load_model_design(ROOT / DESIGN)
    assert list(module.SHARED_PARENT_MAP) == MAPPING
    assert module.SHARED_PREFIX_BLOCKS == 8
    assert module.SNR_EXPERT_EDGES_DB == tuple(range(-20, 21, 5))
    assert module._expert_indices(torch.arange(-20., 21., 5.)).tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 7]
    assert module.NUM_BITS_PER_UE == 1152 and module.NUM_BITS_PER_RE == 8
    assert module.TRANSMITTER_ROUTING == "whole_min"


def test_prefix_shares_suffix_independent_and_optimizer_deduplicates(link):
    assert assert_aliases(link) > 0
    assert sum(p.numel() for p in link.parameters()) < 100_000_000
    assert link.transmitter.experts[0]._out is not link.transmitter.experts[1]._out
    assert link.receiver.experts[2]._norm is not link.receiver.experts[3]._norm


def test_state_dict_contains_all_eight_routes_but_shared_storage(link):
    for component in ("transmitter", "receiver"):
        state = getattr(link, component).state_dict()
        assert {int(key.split(".")[1]) for key in state} == set(range(8))
        for group in ([0, 1], [2, 3, 4, 5, 6, 7]):
            shared = [state[f"experts.{i}._blocks.0._mlp.0.weight"].data_ptr() for i in group]
            private = [state[f"experts.{i}._blocks.8._mlp.0.weight"].data_ptr() for i in group]
            assert len(set(shared)) == 1 and len(set(private)) == len(group)


def test_name_canonicalization_preserves_private_experts():
    assert shared_name("transmitter", "_feedback_expand.0.0.weight")
    assert not shared_name("receiver", "_norm.weight")
    assert canonical_name("receiver", 7, "_blocks.7._scale") == "receiver.experts.2._blocks.7._scale"
    assert canonical_name("receiver", 7, "_blocks.8._scale") == "receiver.experts.7._blocks.8._scale"
    assert canonical_name("transmitter", 1, "_embed.weight") == "transmitter.experts.0._embed.weight"


def test_cpu_conversion_keeps_aliases(link):
    link.to(device="cpu", dtype=torch.float64)
    assert assert_aliases(link) > 0
    link.to(dtype=torch.float32)
    assert assert_aliases(link) > 0


def test_shared_prefix_gradient_sums_and_private_suffix_does_not(link):
    link.zero_grad(set_to_none=True)
    experts = link.receiver.experts
    loss = 2*experts[0]._blocks[0]._scale + 3*experts[1]._blocks[0]._scale
    loss = loss + 7*experts[0]._blocks[8]._scale + 11*experts[1]._blocks[8]._scale
    loss.backward()
    assert experts[0]._blocks[0]._scale.grad.item() == 5
    assert experts[1]._blocks[0]._scale.grad.item() == 5
    assert experts[0]._blocks[8]._scale.grad.item() == 7
    assert experts[1]._blocks[8]._scale.grad.item() == 11
    assert experts[2]._blocks[0]._scale.grad is None

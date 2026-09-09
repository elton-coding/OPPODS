import ast
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import probe_pure_neural_sixteen_v254 as probe


def load_design():
    spec = importlib.util.spec_from_file_location("v254_test", ROOT / probe.DESIGN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_expert_edges_differ_from_v250_design():
    def without_edges(path):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "SNR_EXPERT_EDGES_DB" for target in node.targets))]
        return ast.dump(tree, include_attributes=False)
    assert without_edges(ROOT / probe.DESIGN) == without_edges(ROOT / "research/pure_neural_v230/modelDesign.py")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sixteen_edges_preserve_all_original_boundaries_and_parent_routes(dtype):
    module = load_design()
    edges = torch.tensor(module.SNR_EXPERT_EDGES_DB, dtype=dtype)
    assert module.NUM_EXPERTS == 16 and len(edges) == 17
    assert edges[::2].tolist() == list(range(-20, 21, 5))
    assert module._expert_indices(edges).tolist() == list(range(16)) + [15]
    centers = (edges[:-1] + edges[1:]) / 2
    assert module._expert_indices(centers).tolist() == list(range(16))
    assert [int(value >= -10) for value in centers] == probe.MAPPING
    masses = ((20 - edges[:-1]) / 40).square() - ((20 - edges[1:]) / 40).square()
    torch.testing.assert_close(masses, torch.arange(31, 0, -2, dtype=dtype) / 256, atol=0, rtol=0)


def test_all_sixteen_ids_fit_unchanged_five_control_bits():
    ids = torch.arange(16)
    bits = (ids[:, None] >> torch.arange(5)) & 1
    assert bits.shape == (16, 5)
    assert torch.equal((bits * (2 ** torch.arange(5))).sum(1), ids)


def test_metadata_and_parameter_count_without_allocating_full_weights():
    module = load_design()
    assert module.TRANSMITTER_ROUTING == "whole_min"
    assert module.NUM_BITS_PER_RE == 8 and module.PAYLOAD_BITS == 1152
    assert module.NUM_CTRL == 5 and module.NUM_UL_RE == 96
    with torch.device("meta"):
        components = [module.Encoder(), module.Transmitter(), module.Receiver()]
    assert sum(p.numel() for model in components for p in model.parameters()) == 379906560

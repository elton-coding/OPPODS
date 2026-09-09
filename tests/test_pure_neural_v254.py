import ast
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import probe_pure_neural_sixteen_v254 as probe
import run_pure_neural_sixteen_v254 as runner
from probe_pure_neural_sixteen_memory_v254 import MEMORY_FRACTION, forced_snr


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


def test_full_state_probe_includes_rare_experts_in_each_batch():
    module = load_design()
    snr = forced_snr(torch.device("cpu"))
    assert snr.shape == (100, 2)
    assert torch.bincount(module._expert_indices(snr.amin(1)), minlength=16).tolist() == [7] * 4 + [6] * 12
    assert torch.all(snr[0::2, 1] == 20) and torch.all(snr[1::2, 0] == 20)
    assert MEMORY_FRACTION == .5


def test_full72k_command_keeps_control_protocol_except_registered_factor_and_allocation():
    original, candidate = runner.control_command(), runner.command()
    start = candidate.index("--baseline-expert-map") + 1
    assert candidate[start:start + 16] == list(map(str, probe.MAPPING))
    candidate[start:start + 16] = original[original.index("--baseline-expert-map") + 1:
                                              original.index("--baseline-expert-map") + 9]
    for flag in ("--model-design", "--output-dir", "--gpu-memory-fraction"):
        candidate[candidate.index(flag) + 1] = original[original.index(flag) + 1]
    assert candidate == original
    smoke = runner.command(probe=True)
    assert smoke[smoke.index("--steps") + 1] == "2"
    assert smoke[smoke.index("--validation-samples") + 1] == "2000"
    assert smoke[smoke.index("--output-dir") + 1] != runner.OUTPUT


def test_partial_eight_expert_or_changed_budget_reports_rejected():
    report = {"requested_steps": 72000, "batch_size": 100, "seed": 15240, "learning_rate": 1e-5,
              "loss_kind": "rms_score", "score_temperature": .5, "score_bce_weight": .05,
              "score_fairness_weight": .3, "quantile_bandwidth": .025,
              "train_components": ["transmitter", "receiver"], "validation_samples": 2000,
              "baseline_expert_map": probe.MAPPING, "gpu_memory_fraction": .5,
              "trainable_parameters": 379103168, "variable_payload": False,
              "history": [{"step": step} for step in range(0, 72001, 1000)], "gpu_peak_allocated_bytes": 1}
    runner.require_result(report)
    for change in ({"history": report["history"][:-1]}, {"baseline_expert_map": [0, 0, 1, 1, 1, 1, 1, 1]},
                   {"batch_size": 50}, {"requested_steps": 36000}, {"gpu_peak_allocated_bytes": 0}):
        with pytest.raises(ValueError):
            runner.require_result({**report, **change})
    runner.require_result({**report, "requested_steps": 2, "history": [{"step": 0}, {"step": 2}]}, probe=True)

import ast
import json
import sys
from itertools import pairwise
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_pure_neural_routing_v252 as runner
from train_pure_neural_snr_experts import load_model_design


def design():
    return load_model_design(ROOT / runner.DESIGN)


def test_only_boundaries_change_in_standalone_design():
    def without_edges(path):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SNR_EXPERT_EDGES_DB"
                                                for t in node.targets))]
        return ast.dump(tree)
    assert without_edges(ROOT / runner.DESIGN) == without_edges(ROOT / "research/pure_neural_v230/modelDesign.py")


def test_min_snr_masses_match_registered_parent_preserving_partition():
    module = design()
    edges = module.SNR_EXPERT_EDGES_DB
    masses = [((20-a)/40)**2 - ((20-b)/40)**2 for a, b in pairwise(edges)]
    assert masses == pytest.approx([7/64] * 4 + [9/64] * 4, abs=1e-14)
    assert sum(masses) == pytest.approx(1.)
    assert edges[4] == -10. and len(edges) == 9
    grid = torch.cat([torch.linspace(-20., 20., 10001), torch.tensor(edges)])
    sources = torch.tensor(runner.MAPPING)[module._expert_indices(grid)]
    assert torch.equal(sources, (grid >= -10.).long())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_boundary_convention_and_extremes(dtype):
    module = design()
    boundaries = torch.tensor(module.SNR_EXPERT_BOUNDARIES_DB, dtype=dtype)
    below = torch.nextafter(boundaries, torch.full_like(boundaries, -torch.inf))
    assert module._expert_indices(boundaries).tolist() == list(range(1, 8))
    assert module._expert_indices(below).tolist() == list(range(7))
    assert module._expert_indices(torch.tensor([-20., 20.], dtype=dtype)).tolist() == [0, 7]


def test_all_eight_control_codes_route_receiver_to_same_expert():
    module = design()
    class TransmitterStub(nn.Module):
        def __init__(self, index):
            super().__init__()
            self.index = index

        def forward(self, bits, feedback, snr):
            return torch.full((len(bits[0]), 16, 144), self.index, dtype=torch.complex64), None

    class ReceiverStub(nn.Module):
        def __init__(self, index):
            super().__init__()
            self.index = index

        def forward(self, y, h, control, snr):
            return torch.full((len(y), 1152), self.index, dtype=torch.float32)

    tx = module.Transmitter.__new__(module.Transmitter)
    rx = module.Receiver.__new__(module.Receiver)
    nn.Module.__init__(tx)
    nn.Module.__init__(rx)
    tx.experts = nn.ModuleList([TransmitterStub(i) for i in range(8)])
    rx.experts = nn.ModuleList([ReceiverStub(i) for i in range(8)])
    edges = torch.tensor(module.SNR_EXPERT_EDGES_DB)
    minimum = (edges[:-1] + edges[1:]) / 2
    snr = torch.stack([minimum, torch.full_like(minimum, 20.)])
    signal, control = tx([torch.zeros(8, 1152)] * 2, [torch.zeros(8, 96, dtype=torch.complex64)] * 2, snr)
    assert control.shape == (8, 5) and ((control == 0) | (control == 1)).all()
    assert signal[:, 0, 0].real.tolist() == list(range(8))
    decoded = rx(torch.zeros(8, 2, 144, dtype=torch.complex64),
                 torch.zeros(8, 2, 16, 144, dtype=torch.complex64), control, minimum)
    assert decoded[:, 0].tolist() == list(range(8))


def test_command_keeps_budget_objective_and_original_parent():
    reference = runner.control_command("eight")
    expected = list(reference)
    for flag, value in (("--model-design", runner.DESIGN), ("--output-dir", runner.OUTPUT)):
        expected[expected.index(flag) + 1] = value
    start = expected.index("--baseline-expert-map") + 1
    expected[start:start+8] = list(map(str, runner.MAPPING))
    assert runner.command() == expected
    probe = runner.command(probe=True)
    assert probe[probe.index("--batch-size")+1] == "100"
    assert probe[probe.index("--validation-samples")+1] == "2000"
    assert probe[probe.index("--steps")+1] == "2"
    assert "artifacts/pure_neural_v227/joint_low" in probe


def test_reports_require_all_registered_steps_and_mapping():
    control = json.loads((ROOT / f"benchmarks/{runner.CONTROL}_audit_offset2000.json").read_text(encoding="utf-8"))["training_report"]
    candidate = {**control, "baseline_expert_map": runner.MAPPING}
    runner.require_result(candidate)
    for changes in ({"history": control["history"][:-1]}, {"score_fairness_weight": .5}, {"baseline_expert_map": [0, 1]}):
        with pytest.raises(ValueError):
            runner.require_result({**candidate, **changes})
    probe = {**candidate, "requested_steps": 2, "history": [{"step": 0}, {"step": 2}]}
    runner.require_result(probe, probe=True)
    with pytest.raises(ValueError):
        runner.require_result({**probe, "batch_size": 2}, probe=True)

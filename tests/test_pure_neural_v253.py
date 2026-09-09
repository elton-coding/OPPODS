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
from probe_pure_neural_pair_v253 import DESIGN, MAPPING, REFERENCE, route_cases
from train_pure_neural_snr_experts import load_model_design


def design():
    return load_model_design(ROOT / DESIGN)


def test_learned_modules_and_receiver_identical_to_balanced_min_control():
    reference = ast.parse((ROOT / REFERENCE).read_text(encoding="utf-8"))
    candidate = ast.parse((ROOT / DESIGN).read_text(encoding="utf-8"))
    for name in ("PositionalEncoding", "EncoderCore", "ResidualMLPBlock", "TransmitterCore", "ReceiverCore",
                 "Encoder", "Receiver"):
        old = next(node for node in reference.body if isinstance(node, ast.ClassDef) and node.name == name)
        new = next(node for node in candidate.body if isinstance(node, ast.ClassDef) and node.name == name)
        assert ast.dump(old) == ast.dump(new)
    old_tx = next(node for node in reference.body if isinstance(node, ast.ClassDef) and node.name == "Transmitter")
    new_tx = next(node for node in candidate.body if isinstance(node, ast.ClassDef) and node.name == "Transmitter")
    for name in ("__init__", "initialize_from_baseline"):
        assert ast.dump(next(n for n in old_tx.body if n.name == name)) == ast.dump(next(n for n in new_tx.body if n.name == name))
    old_route = next(n for n in old_tx.body if n.name == "_whole_min_forward")
    new_route = next(n for n in new_tx.body if n.name == "_whole_pair_forward")
    # Apart from deriving the route index, dispatch and control encoding are unchanged.
    assert [ast.dump(n) for n in old_route.body[2:]] == [ast.dump(n) for n in new_route.body[1:]]


def test_probabilities_match_v252_and_preserve_parent_boundary():
    module = design()
    masses = [(((20-a)/40)**2 - ((20-b)/40)**2)/2 for a, b in pairwise(module.SNR_EXPERT_EDGES_DB)]
    masses = [value for value in masses for _ in range(2)]
    assert masses == pytest.approx([7/64]*4 + [9/64]*4, abs=1e-14)
    assert module.NUM_EXPERTS == 8 and module.NUM_CTRL == 5
    grid = torch.linspace(-20, 20, 201)
    a, b = torch.meshgrid(grid, grid, indexing="ij")
    pairs = torch.stack([a.flatten(), b.flatten()])
    routes = module._pair_expert_indices(pairs)
    assert torch.equal(routes, module._pair_expert_indices(pairs.flip(0)))
    assert torch.equal(torch.tensor(MAPPING)[routes], (pairs.amin(0) >= -10).long())
    assert module._pair_expert_indices(torch.tensor([[-20., 20.], [-20., 20.]])).tolist() == [0, 7]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_minimum_and_conditional_maximum_boundary_conventions(dtype):
    module = design()
    boundaries = torch.tensor(module.SNR_EXPERT_BOUNDARIES_DB, dtype=dtype)
    below = torch.nextafter(boundaries, torch.full_like(boundaries, -torch.inf))
    assert module._minimum_bin_indices(boundaries).tolist() == [1, 2, 3]
    assert module._minimum_bin_indices(below).tolist() == [0, 1, 2]
    minimum = torch.tensor([-18., -12., -5., 10.], dtype=dtype)
    cutoff = (minimum + 20)/2
    just_below = torch.nextafter(cutoff, torch.full_like(cutoff, -torch.inf))
    assert module._pair_expert_indices(torch.stack([minimum, cutoff])).tolist() == [1, 3, 5, 7]
    assert module._pair_expert_indices(torch.stack([minimum, just_below])).tolist() == [0, 2, 4, 6]


def test_deterministic_pair_grid_observed_masses_match_analytic_partition():
    module = design()
    grid = -20. + (torch.arange(800, dtype=torch.float64) + .5)*40/800
    a, b = torch.meshgrid(grid, grid, indexing="ij")
    counts = torch.bincount(module._pair_expert_indices(torch.stack([a.flatten(), b.flatten()])), minlength=8)
    assert (counts / counts.sum()).tolist() == pytest.approx([7/64]*4 + [9/64]*4, abs=.0015)


def test_all_eight_joint_codes_match_transmitter_and_receiver():
    module = design()
    class Tx(nn.Module):
        def __init__(self, index):
            super().__init__()
            self.index = index

        def forward(self, bits, feedback, snr):
            return torch.full((len(bits[0]), 16, 144), self.index, dtype=torch.complex64), None

    class Rx(nn.Module):
        def __init__(self, index):
            super().__init__()
            self.index = index

        def forward(self, y, h, control, snr):
            return torch.full((len(y), 1152), self.index, dtype=torch.float32)

    tx, rx = module.Transmitter.__new__(module.Transmitter), module.Receiver.__new__(module.Receiver)
    nn.Module.__init__(tx)
    nn.Module.__init__(rx)
    tx.experts, rx.experts = nn.ModuleList([Tx(i) for i in range(8)]), nn.ModuleList([Rx(i) for i in range(8)])
    snr = route_cases(module).T
    signal, control = tx([torch.zeros(8, 1152)]*2, [torch.zeros(8, 96, dtype=torch.complex64)]*2, snr)
    assert signal[:, 0, 0].real.tolist() == list(range(8))
    assert control.shape == (8, 5) and ((control == 0) | (control == 1)).all()
    for user in range(2):
        output = rx(torch.zeros(8, 2, 144, dtype=torch.complex64),
                    torch.zeros(8, 2, 16, 144, dtype=torch.complex64), control, snr[user])
        assert output[:, 0].tolist() == list(range(8))


def test_training_command_preserves_control_budget_loss_and_parent():
    import run_pure_neural_pair_v253 as runner
    expected = runner.control_command("eight")
    for flag, value in (("--model-design", DESIGN), ("--output-dir", runner.OUTPUT)):
        expected[expected.index(flag)+1] = value
    start = expected.index("--baseline-expert-map") + 1
    expected[start:start+8] = list(map(str, MAPPING))
    assert runner.command() == expected
    probe = runner.command(probe=True)
    for flag, value in (("--steps", "2"), ("--validate-every", "2"), ("--batch-size", "100"),
                        ("--validation-samples", "2000"), ("--baseline-dir", "artifacts/pure_neural_v227/joint_low")):
        assert probe[probe.index(flag)+1] == value
    assert runner.DEPENDENCY == "v250_eight_rms_72k"
    assert runner.REGION_CONTROL == "v252_balanced_routing_rms_36k"


def test_report_guard_rejects_incomplete_budget_wrong_loss_or_mapping():
    import run_pure_neural_pair_v253 as runner
    control = json.loads((ROOT / f"benchmarks/{runner.CONTROL}_audit_offset2000.json").read_text(encoding="utf-8"))["training_report"]
    report = {**control, "baseline_expert_map": MAPPING}
    runner.require_result(report)
    for update in ({"history": report["history"][:-1]}, {"score_fairness_weight": .5}, {"baseline_expert_map": [0, 1]}):
        with pytest.raises(ValueError):
            runner.require_result({**report, **update})


def test_gpu_slot_guard_rejects_third_registered_training_job(monkeypatch):
    from types import SimpleNamespace

    import run_pure_neural_pair_v253 as runner
    cmd = ["python", "train.py", "--stage", "calibrate", "--gpu-memory-fraction", ".4"]
    jobs = [SimpleNamespace(info={"pid": i, "name": "python.exe", "cmdline": cmd}) for i in (1, 2)]
    monkeypatch.setattr(runner.psutil, "process_iter", lambda fields: jobs)
    with pytest.raises(RuntimeError, match="two registered"):
        runner.check_training_slot()
    jobs.pop()
    assert runner.check_training_slot() == [1]

import copy
import json
import sys
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_pure_neural_cosine_v255 as runner
import train_pure_neural_cosine_v255 as cosine


def test_registered_rate_boundaries_and_monotonic_decay():
    assert cosine.learning_rate(1) == cosine.learning_rate(36000) == 1e-5
    assert cosine.learning_rate(36001) < 1e-5
    assert cosine.learning_rate(72000) == pytest.approx(1e-6)
    values = [cosine.learning_rate(i) for i in range(36000, 72001)]
    assert all(a >= b >= 1e-6 for a, b in pairwise(values))


@pytest.mark.parametrize("step", [0, -1, 72001, 1.5])
def test_outside_registered_schedule_rejected(step):
    with pytest.raises(ValueError):
        cosine.learning_rate(step)


def test_hooks_preserve_adam_state_and_rng_in_constant_prefix():
    a, b = (torch.nn.Parameter(torch.tensor([.1, -.2])) for _ in range(2))
    oa, ob = torch.optim.Adam([a], lr=1e-5), torch.optim.Adam([b], lr=1e-5)
    recorder = cosine.RateRecorder(ob)
    state = torch.random.get_rng_state().clone()
    for step in range(1, 21):
        a.grad = torch.tensor([step * .01, -.1])
        b.grad = a.grad.clone()
        oa.step()
        ob.step()
        assert torch.equal(a, b)
        for key in oa.state[a]:
            assert torch.equal(oa.state[a][key], ob.state[b][key])
    assert torch.equal(state, torch.random.get_rng_state())
    cosine.require_schedule(recorder.evidence(), 20)


def test_decay_is_applied_before_actual_update_without_resetting_adam():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.Adam([parameter], lr=1e-5)
    recorder = cosine.RateRecorder(optimizer)
    parameter.grad = torch.ones(1)
    optimizer.step()
    moment = optimizer.state[parameter]["exp_avg"]
    recorder.completed = 71999  # A targeted hook boundary test, not full trajectory evidence.
    optimizer.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-6)
    assert optimizer.state[parameter]["exp_avg"] is moment
    assert optimizer.state[parameter]["step"].item() == 2
    with pytest.raises(ValueError):
        cosine.require_schedule(recorder.evidence(), 72000)


def test_partial_update_or_mutated_rate_cannot_be_certified():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.Adam([parameter], lr=1e-5)
    recorder = cosine.RateRecorder(optimizer)
    recorder.before(optimizer, (), {})
    with pytest.raises(RuntimeError):
        recorder.evidence()
    optimizer.param_groups[0]["lr"] = 2e-5
    with pytest.raises(RuntimeError):
        recorder.after(optimizer, (), {})


def test_evidence_digest_and_update_count_are_mandatory():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.Adam([parameter], lr=1e-5)
    recorder = cosine.RateRecorder(optimizer)
    parameter.grad = torch.ones(1)
    optimizer.step()
    evidence = recorder.evidence()
    cosine.require_schedule(evidence, 1)
    changed = copy.deepcopy(evidence)
    changed["applied_rates_float64_le_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        cosine.require_schedule(changed, 1)
    with pytest.raises(ValueError):
        cosine.require_schedule(evidence, 2)


def test_formal_command_changes_only_entry_and_output_not_training_factors():
    expected = runner.control_command()
    expected[2] = "scripts/train_pure_neural_cosine_v255.py"
    expected[expected.index("--output-dir") + 1] = runner.OUTPUT
    assert runner.command() == expected
    probe = runner.command(probe=True)
    assert probe[probe.index("--steps") + 1] == "2"
    assert probe[probe.index("--batch-size") + 1] == "100"
    assert probe[probe.index("--validation-samples") + 1] == "2000"


def test_full_report_requires_actual_schedule_and_all_checkpoints():
    audit = json.loads((ROOT / "benchmarks/v250_eight_rms_72k_audit_offset2000.json").read_text(encoding="utf-8"))
    report = audit["training_report"]
    with pytest.raises(ValueError):
        runner.require_result(report)
    proof = json.loads((ROOT / "benchmarks/v255_cosine_cpu_proof.json").read_text(encoding="utf-8"))
    report = copy.deepcopy(report)
    report["learning_rate_schedule"] = proof["actual_schedule"]
    report["learning_rate_field_scope"] = "initial rate only; actual applied rates bound by learning_rate_schedule"
    runner.require_result(report)  # Synthetic report validation fixture, not a completed V255 model.
    report["history"].pop()
    with pytest.raises(ValueError):
        runner.require_result(report)


@pytest.mark.parametrize("fail", [False, True])
def test_entry_hooks_are_process_local_and_restored(monkeypatch, tmp_path, fail):
    args = SimpleNamespace(stage="calibrate", loss_kind="rms_score", steps=2, learning_rate=1e-5,
                           train_components=["transmitter", "receiver"], optimize_expert_index=None,
                           batch_size=100, output_dir=tmp_path / "output")
    monkeypatch.setattr(cosine.rms, "parse_args", lambda: args)
    original_adam = torch.optim.Adam
    original_parse = cosine.rms.trainer.parse_args
    original_loss = cosine.rms.trainer.score_aligned_loss

    def tiny_trainer():
        assert cosine.rms.trainer.parse_args() is args
        if fail:
            raise RuntimeError("deliberate trainer failure")
        parameter = torch.nn.Parameter(torch.tensor([.1, -.2]))
        optimizer = torch.optim.Adam([parameter], lr=1e-5)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            parameter.square().sum().backward()
            optimizer.step()
        args.output_dir.mkdir()
        (args.output_dir / "training_report.json").write_text(json.dumps({"history": [{"step": 0}, {"step": 2}]}), encoding="utf-8")

    monkeypatch.setattr(cosine.rms.trainer, "main", tiny_trainer)
    if fail:
        with pytest.raises(RuntimeError, match="deliberate"):
            cosine.main()
    else:
        cosine.main()
        report = json.loads((args.output_dir / "training_report.json").read_text(encoding="utf-8"))
        cosine.require_schedule(report["learning_rate_schedule"], 2)
    assert torch.optim.Adam is original_adam
    assert cosine.rms.trainer.parse_args is original_parse
    assert cosine.rms.trainer.score_aligned_loss is original_loss

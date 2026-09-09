import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_pure_neural_encoder_lr_v249 as runner
import train_pure_neural_encoder_lr_v249 as entry
from run_pure_neural_lr_v237 import train_command


def test_command_matches_v235_except_optimizer_entry():
    expected = train_command(runner.PLAN)
    expected.insert(expected.index("--train-components") + 1, "encoder")
    actual = runner.command()
    assert actual[2] == "scripts/train_pure_neural_encoder_lr_v249.py"
    actual[2] = expected[2]
    assert actual == expected


def test_grouped_adam_changes_only_encoder_update_scale():
    params = [torch.nn.Parameter(torch.tensor([1., -2.], dtype=torch.float64)) for _ in range(3)]
    baseline = [torch.nn.Parameter(p.detach().clone()) for p in params]
    original = [p.detach().clone() for p in params]
    groups = entry.parameter_groups(params, params[:1], 1e-5)
    assert [id(p) for group in groups for p in group["params"]] == [id(p) for p in params]
    candidate = torch.optim.Adam(groups, lr=1e-5)
    control = torch.optim.Adam(baseline, lr=1e-5)
    for p, q in zip(params, baseline, strict=True):
        p.grad = torch.tensor([.3, -.7], dtype=p.dtype)
        q.grad = p.grad.clone()
    candidate.step()
    control.step()
    torch.testing.assert_close(original[0] - params[0], 5 * (original[0] - baseline[0]), atol=1e-14, rtol=1e-10)
    for p, q in zip(params[1:], baseline[1:], strict=True):
        torch.testing.assert_close(p, q, atol=0, rtol=0)


def test_grouping_rejects_duplicates_and_missing_encoder():
    p, q, absent = [torch.nn.Parameter(torch.ones(1)) for _ in range(3)]
    for parameters, encoder in (([p, p, q], [p]), ([p, q], []), ([p, q], [absent]), ([p, q], [p, q])):
        with pytest.raises(ValueError):
            entry.parameter_groups(parameters, encoder, 1e-5)


def test_entry_restores_hooks_after_training_failure(monkeypatch, tmp_path):
    args = SimpleNamespace(stage="calibrate", train_components=["encoder", "transmitter", "receiver"],
                           learning_rate=1e-5, optimize_expert_index=None, output_dir=tmp_path / "new")
    parse = lambda: args
    monkeypatch.setattr(entry.trainer, "parse_args", parse)
    original_select, original_adam = entry.trainer.select_trainable_parameters, torch.optim.Adam
    def fail():
        assert torch.optim.Adam is not original_adam
        raise RuntimeError("intentional training failure")
    monkeypatch.setattr(entry.trainer, "main", fail)
    with pytest.raises(RuntimeError, match="intentional"):
        entry.main()
    assert entry.trainer.parse_args is parse
    assert entry.trainer.select_trainable_parameters is original_select
    assert torch.optim.Adam is original_adam


def test_report_requires_group_metadata_and_full_budget():
    report = {"requested_steps": 12000, "train_components": ["encoder", "transmitter", "receiver"],
              "trainable_parameters": 48191288, "batch_size": 100, "learning_rate": 1e-5,
              "encoder_learning_rate": 5e-5, "transceiver_learning_rate": 1e-5,
              "baseline_expert_map": [0, 1], "seed": 15240, "loss_kind": "soft_score",
              "score_temperature": .5, "score_bce_weight": .05, "score_fairness_weight": .3,
              "quantile_bandwidth": .025, "validation_samples": 2000,
              "history": [{"step": 12000}], "gpu_peak_allocated_bytes": 1,
              "optimizer_parameter_groups": [{"name": "encoder", "lr": 5e-5, "parameters": 803392},
                                              {"name": "transceiver", "lr": 1e-5, "parameters": 47387896}]}
    runner.require_report(report)
    for change in ({"history": [{"step": 11000}]}, {"optimizer_parameter_groups": []}, {"encoder_learning_rate": 1e-5}):
        with pytest.raises(ValueError):
            runner.require_report({**report, **change})

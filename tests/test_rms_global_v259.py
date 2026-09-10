import copy
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_pure_neural_rms_global_v259 as runner
import train_pure_neural_rms_global_v259 as entry
from probe_pure_neural_rms_global_v259 import objective


def fixture_report():
    report = copy.deepcopy(json.loads((ROOT / "benchmarks/v250_eight_rms_72k_audit_offset2000.json").read_text())["training_report"])
    report.update(requested_steps=18000, batch_size=400, validation_batch_size=100, microbatch_size=100,
                  global_loss_ue_count=800, training_channel_draws=7200000,
                  activation_checkpointing="nonreentrant; explicit per-chunk generator replay")
    report["history"] = [{**row, "step": row["step"] // 4} for row in report["history"]]
    report["best_step"] //= 4
    return report


def test_command_matches_channel_budget_without_other_factors():
    expected = runner.control_command()
    expected[2] = "scripts/train_pure_neural_rms_global_v259.py"
    for flag, value in (("--batch-size", "400"), ("--steps", "18000"), ("--validate-every", "250"),
                        ("--output-dir", runner.OUTPUT)):
        expected[expected.index(flag) + 1] = value
    assert runner.command() == expected + ["--microbatch-size", "100"]
    assert 400 * 18000 == 100 * 72000
    assert 400 * 250 == 100 * 1000
    probe = runner.command(probe=True)
    assert probe[probe.index("--steps") + 1] == "2"
    assert probe[probe.index("--validation-samples") + 1] == "2000"


def test_complete_report_fixtures_and_no_mutation():
    report = fixture_report()  # Schema fixture, not a V259 score/training claim.
    before = copy.deepcopy(report)
    runner.require_result(report)
    assert report == before
    report.update(requested_steps=2, training_channel_draws=800)
    report["history"] = [report["history"][0], {**report["history"][1], "step": 2}]
    runner.require_result(report, probe=True)


@pytest.mark.parametrize("field,value", [
    ("batch_size", 100), ("global_loss_ue_count", 200), ("microbatch_size", 50),
    ("validation_batch_size", 400), ("training_channel_draws", 1200000), ("loss_kind", "soft_score"),
    ("score_temperature", .25), ("learning_rate", 5e-5), ("learning_rate_schedule", {}),
    ("train_components", ["encoder", "transmitter", "receiver"]),
])
def test_other_factors_or_wrong_global_loss_metadata_rejected(field, value):
    report = fixture_report()
    report[field] = value
    with pytest.raises(ValueError):
        runner.require_result(report)


def test_partial_history_or_changed_evaluation_schedule_rejected():
    report = fixture_report()
    report["history"][35]["step"] += 1
    with pytest.raises(ValueError):
        runner.require_result(report)


def test_global_rms_quantile_is_not_average_of_microbatch_quantiles():
    generator = torch.Generator().manual_seed(259)
    logits = torch.randn(16, 2, 1152, generator=generator, dtype=torch.float64)
    logits += torch.linspace(-2., 2., 16)[:, None, None]
    bits = torch.zeros_like(logits)
    whole = objective(logits, bits)
    separate = torch.stack([objective(logits[i:i+4], bits[i:i+4]) for i in range(0, 16, 4)]).mean()
    assert abs(float(whole - separate)) > 1e-6


def test_composed_entry_uses_one_rms_loss_and_original_validation_then_restores(monkeypatch, tmp_path):
    global_entry, trainer = entry.global_entry, entry.global_entry.trainer
    output = tmp_path / "probe"
    monkeypatch.setattr(sys, "argv", ["v259", "--stage", "calibrate", "--loss-kind", "rms_score",
                                    "--output-dir", str(output), "--batch-size", "400", "--steps", "2",
                                    "--microbatch-size", "100"])
    original = (global_entry.parse_args, trainer.parse_args, trainer.PureNeuralLink.forward,
                trainer.evaluate, trainer.score_aligned_loss)
    calls = []
    real_loss = entry.rms_score_loss

    def recording_loss(logits, bits, **options):
        calls.append((tuple(logits.shape), options["loss_kind"]))
        return real_loss(logits, bits, **options)

    def forward(link, channel, bits, snr, *, generator, shared_frontend=False):
        return link(channel + torch.randn(channel.shape, generator=generator)).sin()

    monkeypatch.setattr(entry, "rms_score_loss", recording_loss)
    monkeypatch.setattr(global_entry, "BASE_FORWARD", forward)
    monkeypatch.setattr(global_entry, "BASE_EVALUATE", lambda **kwargs: kwargs["batch_size"])

    def simulate_training():
        args = trainer.parse_args()
        assert args.loss_kind == "rms_score" and args.batch_size == 400
        assert trainer.evaluate(batch_size=400) == 100
        model = torch.nn.Linear(8, 8)
        values = torch.randn(400, 2, 8)
        bits = torch.zeros_like(values)
        generator = torch.Generator().manual_seed(259)
        logits = trainer.PureNeuralLink.forward(model, values, bits, torch.zeros(400, 2), generator=generator)
        state = generator.get_state().clone()
        loss = trainer.score_aligned_loss(logits, bits, loss_kind=args.loss_kind, margin=.5, tail_weight=0.,
                                         tail_fraction=.1, score_temperature=.5, quantile_bandwidth=.025,
                                         score_bce_weight=.05, score_fairness_weight=.3, valid_lengths=None)
        loss.backward()
        assert torch.equal(state, generator.get_state())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        output.mkdir()
        (output / "training_report.json").write_text(json.dumps({"history": [{"step": 0}, {"step": 2}],
                                                                "loss_kind": args.loss_kind}))

    monkeypatch.setattr(trainer, "main", simulate_training)
    entry.main()
    assert calls == [((400, 2, 8), "rms_score")]
    report = json.loads((output / "training_report.json").read_text())
    assert report["global_loss_ue_count"] == 800 and report["training_channel_draws"] == 800
    assert report["validation_batch_size"] == 100 and report["loss_kind"] == "rms_score"
    assert original == (global_entry.parse_args, trainer.parse_args, trainer.PureNeuralLink.forward,
                        trainer.evaluate, trainer.score_aligned_loss)


def test_entry_restores_loss_and_parser_after_failure(monkeypatch):
    original = (entry.global_entry.parse_args, entry.global_entry.trainer.score_aligned_loss)

    def fail():
        assert entry.global_entry.trainer.score_aligned_loss is entry.rms_score_loss
        raise RuntimeError("deliberate")

    monkeypatch.setattr(entry.global_entry, "main", fail)
    with pytest.raises(RuntimeError, match="deliberate"):
        entry.main()
    assert original == (entry.global_entry.parse_args, entry.global_entry.trainer.score_aligned_loss)


def test_runtime_requires_global_counts_frozen_encoder_and_preserved_rng(monkeypatch):
    record = json.loads((ROOT / runner.CPU_PROOF).read_text())
    monkeypatch.setattr(runner, "verify_bound_inputs", lambda _: None)
    runner.require_runtime(record, "cpu")
    for field, value in (("global_loss_ue_count", 16), ("adam_parameter_states", 1200),
                         ("backward_preserves_noise_rng", False), ("loss", float("nan"))):
        bad = copy.deepcopy(record)
        bad["steps"][0][field] = value
        with pytest.raises(ValueError):
            runner.require_runtime(bad, "cpu")
    with pytest.raises(ValueError):
        runner.require_runtime(record, "cuda")


def test_wait_rejects_missing_partial_and_wrong_dependency(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    with pytest.raises(TimeoutError):
        runner.wait_dependency(0)
    path = tmp_path / f"benchmarks/{runner.DEPENDENCY}_audit_offset2000.json"
    path.parent.mkdir()
    path.write_text("{", encoding="utf-8")
    with pytest.raises(TimeoutError):
        runner.wait_dependency(0)
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        runner.wait_dependency(0)

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_pure_neural_global_batch_v246 as batch_trainer


def test_fixed_channel_budget_and_full_batch_probe():
    from run_pure_neural_global_batch_v246 import command, require_report
    args = command()
    values = lambda flag: args[args.index(flag) + 1]
    assert int(values("--batch-size")) * int(values("--steps")) == 1200000
    assert values("--validate-every") == "250"
    assert values("--microbatch-size") == "100"
    assert values("--score-bce-weight") == "0.05"
    probe = command(probe=True)
    assert probe[probe.index("--batch-size") + 1] == "400"
    with pytest.raises(ValueError, match="incomplete"):
        require_report({"requested_steps": 3000, "history": [{"step": 2999}]})


def test_checkpoint_replays_explicit_noise_and_keeps_global_quantile_gradient(monkeypatch):
    torch.manual_seed(246)
    model = torch.nn.Linear(8, 8).double()
    values = torch.randn(12, 2, 8, dtype=torch.float64)
    bits = torch.randint(0, 2, values.shape).double()
    snr = torch.zeros(12, 2)

    def forward(link, channel, bits, snr, *, generator, shared_frontend=False):
        noise = torch.randn(channel.shape, generator=generator, dtype=channel.dtype)
        return link(channel + noise).sin()

    monkeypatch.setattr(batch_trainer, "BASE_FORWARD", forward)
    results = []
    for recompute in (False, True):
        model.zero_grad(set_to_none=True)
        generator = torch.Generator().manual_seed(123)
        logits = batch_trainer.chunked_forward(model, values, bits, snr, generator=generator,
                                               microbatch=5, recompute=recompute)
        end_state = generator.get_state().clone()
        loss = batch_trainer.trainer.score_aligned_loss(
            logits, bits, loss_kind="soft_score", margin=.5, tail_weight=0., tail_fraction=.1,
            score_temperature=.5, quantile_bandwidth=.025, score_fairness_weight=.3, score_bce_weight=.05)
        loss.backward()
        assert torch.equal(generator.get_state(), end_state)
        results.append((logits.detach(), loss.detach(), [p.grad.clone() for p in model.parameters()]))
    for actual, expected in zip(results[0][:2], results[1][:2], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(results[0][2], results[1][2], strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_replay_generator_restores_after_failure():
    generator = torch.Generator().manual_seed(1)
    state = generator.get_state().clone()
    alternate = torch.Generator().manual_seed(2).get_state()
    with pytest.raises(RuntimeError), batch_trainer.replay_generator(generator, alternate):
        assert torch.equal(generator.get_state(), alternate)
        raise RuntimeError("deliberate")
    assert torch.equal(generator.get_state(), state)


def test_validation_batch_is_fixed_and_entry_restores_on_error(monkeypatch, tmp_path):
    monkeypatch.setattr(batch_trainer, "BASE_EVALUATE", lambda **kwargs: kwargs["batch_size"])
    assert batch_trainer.evaluate(batch_size=400) == 100
    monkeypatch.setattr(sys, "argv", ["v246", "--stage", "calibrate", "--loss-kind", "soft_score",
                                    "--output-dir", str(tmp_path / "new"), "--microbatch-size", "100"])
    original = (batch_trainer.trainer.parse_args, batch_trainer.trainer.PureNeuralLink.forward,
                batch_trainer.trainer.evaluate)

    def fail():
        raise RuntimeError("deliberate")

    monkeypatch.setattr(batch_trainer.trainer, "main", fail)
    with pytest.raises(RuntimeError, match="deliberate"):
        batch_trainer.main()
    assert original == (batch_trainer.trainer.parse_args, batch_trainer.trainer.PureNeuralLink.forward,
                        batch_trainer.trainer.evaluate)

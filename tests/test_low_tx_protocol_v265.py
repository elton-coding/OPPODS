import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_pure_neural_low_tx_untie_v265 as r


def fixture():
    a = json.loads((ROOT / f"benchmarks/{r.CONTROL}_audit_offset2000.json").read_text(encoding="utf-8"))
    result = copy.deepcopy(a["training_report"])
    result["trainable_parameters"] = 83767368
    return result


def test_only_architecture_and_output_changed():
    expected = r.control_command()
    expected[expected.index("--model-design") + 1] = r.DESIGN
    expected[expected.index("--output-dir") + 1] = r.OUTPUT
    assert r.command() == expected
    probe = r.command(probe=True)
    assert probe[probe.index("--steps") + 1] == "2"
    assert probe[probe.index("--batch-size") + 1] == "100"


@pytest.mark.parametrize("key,value", [("quantile_bandwidth", .05), ("score_bce_weight", 0.),
    ("score_fairness_weight", .5), ("score_temperature", .25), ("trainable_parameters", 73547840),
    ("requested_steps", 71000), ("learning_rate_schedule", {}), ("microbatch_size", 25)])
def test_guard_rejects_extra_factors(key, value):
    report = fixture()
    r.require_result(report)  # Synthetic schema only, not a training result.
    report[key] = value
    with pytest.raises(ValueError):
        r.require_result(report)


def test_partial_and_nonfinite_rejected():
    report = fixture()
    report["history"].pop()
    with pytest.raises(ValueError):
        r.require_result(report)
    report = fixture()
    report["history"][1]["final"] = float("nan")
    with pytest.raises(ValueError):
        r.require_result(report)


def test_selection_gate():
    comparison = {"per_seed": [{"noise_seed": seed, "delta": {"final": v}}
                               for seed, v in zip(r.SEEDS, [.01, .02, .03], strict=True)],
                  "paired_delta_95_percentile_interval": [.001, .05]}
    assert r.decision(comparison)["eligible_for_new_confirmation"]
    comparison["per_seed"][0]["delta"]["final"] = -.001
    assert not r.decision(comparison)["eligible_for_new_confirmation"]
    comparison["per_seed"][0]["delta"]["final"] = .01
    comparison["paired_delta_95_percentile_interval"][0] = -.001
    assert not r.decision(comparison)["eligible_for_new_confirmation"]


def test_actual_control_metrics_and_tampering():
    audit = json.loads((ROOT / f"benchmarks/{r.CONTROL}_audit_offset2000.json").read_text(encoding="utf-8"))
    cached = r.checked_caches([r.CONTROL], offset=2000, seeds=r.SEEDS)[r.CONTROL]
    r.require_metrics(audit, cached)
    audit["per_seed"][0]["final"] += .1
    with pytest.raises(ValueError):
        r.require_metrics(audit, cached)


def test_existing_output_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(r, "ROOT", tmp_path)
    (tmp_path / r.OUTPUT).mkdir(parents=True)
    with pytest.raises(FileExistsError):
        r.main()

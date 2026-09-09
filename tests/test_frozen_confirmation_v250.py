import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_frozen_confirmation_v250 as runner


def valid_audit():
    return {"label": "v250_confirm", "protocol": copy.deepcopy(runner.PROTOCOL), "exact_mean_final": 68.7,
            "comparisons": {"v242e_confirm_v250": {
                "per_seed": [{"noise_seed": seed, "delta": {"final": .1, "efficiency": .1, "p10": -.02}}
                             for seed in runner.SEEDS],
                "paired_delta_95_percentile_interval": [.02, .15], "mean_delta": .1}}}


def test_new_window_fixed_sources_and_full_budgets():
    assert runner.OFFSET == 6000
    assert runner.SEEDS == [42701, 42702, 42703]
    assert [(row[0], row[3]) for row in runner.SOURCES] == [
        ("v242_eight_rms_36k", 36000), ("v250_eight_rms_72k", 72000)]
    cmd = runner.command("baseline", "frozen", "")
    assert cmd[cmd.index("--test-offset") + 1] == "6000"
    assert cmd[cmd.index("--baseline-label") + 1] == ""
    assert cmd[-3:] == ["42701", "42702", "42703"]


def test_conditional_total_score_gate_preserves_fairness_tradeoff_and_caveats():
    result = runner.decision(valid_audit())
    assert result["supports_current_cohort_promotion"]
    assert result["per_seed_metric_deltas"][0]["p10"] == -.02
    assert not result["whole_project_blindness_certified"]
    assert not result["ancestor_training_provenance_certified"]
    assert not result["online_confirmation"]


@pytest.mark.parametrize("field,value", [("final", -.01), ("final", 0), ("interval", -.01), ("interval", 0)])
def test_nonpositive_improvement_rejected(field, value):
    audit = valid_audit()
    comparison = audit["comparisons"]["v242e_confirm_v250"]
    if field == "final":
        comparison["per_seed"][0]["delta"]["final"] = value
    else:
        comparison["paired_delta_95_percentile_interval"][0] = value
    assert not runner.decision(audit)["supports_current_cohort_promotion"]


@pytest.mark.parametrize("case", ["window", "duplicate_seed", "missing_seed", "nan", "reversed_interval"])
def test_invalid_confirmation_rejected(case):
    audit = valid_audit()
    comparison = audit["comparisons"]["v242e_confirm_v250"]
    if case == "window":
        audit["protocol"]["test_offset"] = 4000
    elif case == "duplicate_seed":
        comparison["per_seed"][1]["noise_seed"] = runner.SEEDS[0]
    elif case == "missing_seed":
        comparison["per_seed"].pop()
    elif case == "nan":
        comparison["per_seed"][0]["delta"]["final"] = float("nan")
    else:
        comparison["paired_delta_95_percentile_interval"] = [.2, .1]
    with pytest.raises(ValueError):
        runner.decision(audit)

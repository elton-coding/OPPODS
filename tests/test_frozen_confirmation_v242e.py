import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_frozen_confirmation_v242e as runner


def test_fixed_window_and_exactly_two_frozen_sources():
    assert runner.OFFSET == 4000
    assert runner.SEEDS == [32701, 32702, 32703]
    assert [source[0] for source in runner.SOURCES] == ["v240_two_36k", "v242_eight_rms_36k"]
    cmd = runner.command("baseline", "frozen", "")
    assert cmd[cmd.index("--baseline-label") + 1] == ""
    assert cmd[cmd.index("--test-offset") + 1] == "4000"


def test_confirmation_requires_all_totals_and_positive_interval():
    comparison = {"per_seed": [{"delta": {"final": .1}} for _ in range(3)],
                  "paired_delta_95_percentile_interval": [.02, .15], "mean_delta": .1}
    audit = {"label": "v242e_confirm", "protocol": {"split_seed": 1176, "test_offset": 4000,
             "samples": 2000, "noise_seeds": runner.SEEDS}, "exact_mean_final": 68.6,
             "comparisons": {"v240c_confirm_v242e": comparison}}
    result = runner.decision(audit)
    assert result["supports_current_cohort_promotion"]
    assert not result["whole_project_blindness_certified"]
    assert not result["ancestor_training_provenance_certified"]
    assert not result["online_confirmation"]
    comparison["per_seed"][0]["delta"]["final"] = -.01
    assert not runner.decision(audit)["supports_current_cohort_promotion"]
    comparison["per_seed"][0]["delta"]["final"] = .1
    comparison["paired_delta_95_percentile_interval"][0] = -.01
    assert not runner.decision(audit)["supports_current_cohort_promotion"]
    audit["protocol"]["test_offset"] = 6000
    with pytest.raises(ValueError):
        runner.decision(audit)

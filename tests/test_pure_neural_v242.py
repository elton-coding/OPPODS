import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_pure_neural_rms_budget_v242 as runner
from run_pure_neural_rms_eight_v241 import command as eight_short
from run_pure_neural_rms_v239 import command as two_short


@pytest.mark.parametrize("arm,short", [("two", two_short), ("eight", eight_short)])
def test_rms_budget_changes_only_registered_budget_and_output(arm, short):
    expected = short()
    for flag, value in (("--steps", "36000"), ("--patience", "36"),
                        ("--output-dir", runner.plan(arm)["output"])):
        expected[expected.index(flag) + 1] = value
    assert runner.command(arm) == expected
    assert runner.plan(arm)["parent"] == "artifacts/pure_neural_v227/joint_low"


def test_rms_budget_audit_requires_exact_completion_and_fixed_protocol():
    report = {"label": "v240_eight_36k", "protocol": dict(runner.PROTOCOL),
              "per_seed": [{}, {}, {}],
              "training_report": {"requested_steps": 36000, "history": [{"step": 36000}]}}
    runner.require_audit(report, "v240_eight_36k", 36000)
    with pytest.raises(ValueError, match="complete"):
        runner.require_audit(report, "v240_eight_36k", 12000)
    report["protocol"]["test_offset"] = 0
    with pytest.raises(ValueError, match="complete"):
        runner.require_audit(report, "v240_eight_36k", 36000)

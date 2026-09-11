import copy

import pytest
import run_shared_lr_v268 as v


def test_only_learning_rate_and_output_change():
    for probe in (False, True):
        candidate, baseline = v.command(probe), v.baseline_command(probe=probe)
        for flag in ("--learning-rate", "--output-dir"):
            i = baseline.index(flag)+1
            candidate[i] = baseline[i]
        assert candidate == baseline


def test_guard_accepts_only_registered_factor_and_complete_budget():
    for probe, path in ((True, "artifacts/resource_probe/v257/eight_shared8/training_report.json"),
                        (False, "artifacts/pure_neural_v257/eight/shared8_steps72000/training_report.json")):
        report = copy.deepcopy(v.read(v.ROOT / path))
        report["learning_rate"] = 2e-5
        v.require_result(report, probe)
        for key, value in (("learning_rate", 1e-5), ("learning_rate_schedule", {}),
                           ("trainable_parameters", 83767368), ("score_bce_weight", 0.),
                           ("microbatch_size", 10), ("requested_steps", 1)):
            with pytest.raises(ValueError):
                v.require_result({**report, key: value}, probe)
        report["history"] = report["history"][:-1]
        with pytest.raises(ValueError):
            v.require_result(report, probe)

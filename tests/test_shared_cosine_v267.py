import copy

import pytest
import run_shared_cosine_v267 as v
import torch
from train_pure_neural_cosine_v255 import RateRecorder


def test_only_schedule_entry_and_output_change():
    for probe in (False, True):
        candidate, baseline = v.command(probe), v.baseline_command(probe=probe)
        candidate[2] = baseline[2]
        i = baseline.index("--output-dir")+1
        candidate[i] = baseline[i]
        assert candidate == baseline


def probe_report():
    report = copy.deepcopy(v.read(v.ROOT / "artifacts/resource_probe/v257/eight_shared8/training_report.json"))
    p = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.Adam([p], lr=1e-5)
    recorder = RateRecorder(optimizer)
    for _ in range(2):
        p.grad = torch.ones_like(p)
        optimizer.step()
    report["learning_rate_schedule"] = recorder.evidence()
    report["learning_rate_field_scope"] = "initial rate only; actual applied rates bound by learning_rate_schedule"
    return report


def test_schedule_and_architecture_guard():
    report = probe_report()
    v.require_result(report, True)
    for key, value in (("trainable_parameters", 83767368), ("score_bce_weight", 0.),
                       ("microbatch_size", 10), ("requested_steps", 1)):
        changed = {**report, key: value}
        with pytest.raises(ValueError):
            v.require_result(changed, True)
    report["learning_rate_schedule"]["completed_updates"] = 1
    with pytest.raises(ValueError):
        v.require_result(report, True)

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_frozen_confirmation_v257 as runner


def valid_audit():
    return {"label": "v257_confirm", "protocol": copy.deepcopy(runner.PROTOCOL), "exact_mean_final": 68.7,
            "comparisons": {"v250_confirm_v257": {
                "per_seed": [{"noise_seed": seed, "delta": {"final": .1, "efficiency": .1, "p10": -.02}}
                             for seed in runner.SEEDS],
                "paired_delta_95_percentile_interval": [.02, .15], "mean_delta": .1}}}


def test_protocol_fixed_new_window_and_paired_sources():
    assert runner.OFFSET == 8000
    assert runner.SEEDS == [52701, 52702, 52703]
    assert [row[0] for row in runner.SOURCES] == ["v250_eight_rms_72k", "v257_eight_shared_prefix8_rms_72k"]
    cmd = runner.command("control", "frozen", "")
    assert cmd[cmd.index("--test-offset")+1] == "8000"
    assert cmd[cmd.index("--baseline-label")+1] == ""
    assert cmd[-3:] == ["52701", "52702", "52703"]


def test_positive_gate_is_conditional_not_automatic_or_online():
    result = runner.decision(valid_audit())
    assert result["supports_current_cohort_promotion"]
    assert result["per_seed_metric_deltas"][0]["p10"] == -.02
    for key in ("automatic_promotion", "whole_project_blindness_certified",
                "ancestor_training_provenance_certified", "online_confirmation"):
        assert result[key] is False


@pytest.mark.parametrize("field,value", [("final", -.01), ("final", 0), ("interval", -.01), ("interval", 0)])
def test_nonpositive_gate_rejected(field, value):
    audit = valid_audit()
    comparison = audit["comparisons"]["v250_confirm_v257"]
    if field == "final":
        comparison["per_seed"][0]["delta"]["final"] = value
    else:
        comparison["paired_delta_95_percentile_interval"][0] = value
    assert not runner.decision(audit)["supports_current_cohort_promotion"]


@pytest.mark.parametrize("case", ["window", "label", "duplicate", "missing", "nan", "reversed", "short_interval"])
def test_invalid_statistics_rejected(case):
    audit = valid_audit()
    comparison = audit["comparisons"]["v250_confirm_v257"]
    if case == "window":
        audit["protocol"]["test_offset"] = 6000
    elif case == "label":
        audit["label"] = "wrong"
    elif case == "duplicate":
        comparison["per_seed"][1]["noise_seed"] = runner.SEEDS[0]
    elif case == "missing":
        comparison["per_seed"].pop()
    elif case == "nan":
        comparison["mean_delta"] = float("nan")
    elif case == "reversed":
        comparison["paired_delta_95_percentile_interval"] = [.2, .1]
    else:
        comparison["paired_delta_95_percentile_interval"] = [.1]
    with pytest.raises(ValueError):
        runner.decision(audit)


def inventory():
    return {"unreadable_archives": 0, "windows": [{"test_offset": 8000, "samples": 2000,
            "train1176_overlap": 0, "validation1176_overlap": 0, "current_offset2000_audit_overlap": 0,
            "explicit_split1176_recorded_ids_overlap": 0, "all_recorded_evaluation_ids_overlap": 1812}]}


def test_historical_overlap_is_disclosed_not_mislabeled_as_blind():
    assert runner.checked_window(inventory())["all_recorded_evaluation_ids_overlap"] == 1812


@pytest.mark.parametrize("field", ["train1176_overlap", "validation1176_overlap",
                                 "current_offset2000_audit_overlap", "explicit_split1176_recorded_ids_overlap"])
def test_current_protocol_reuse_rejected(field):
    record = inventory()
    record["windows"][0][field] = 1
    with pytest.raises(ValueError):
        runner.checked_window(record)


def test_unreadable_inventory_rejected():
    record = inventory()
    record["unreadable_archives"] = 1
    with pytest.raises(ValueError):
        runner.checked_window(record)


def test_pending_parent_is_resource_busy_but_confirmation_not_self_blocking():
    for name in runner.QUEUE_RUNNERS:
        assert runner.is_queue_runner("python.exe", ["python.exe", "-u", f"D:\\Source\\OPPODS\\scripts\\{name}"])
    assert not runner.is_queue_runner("powershell.exe", ["run_pure_neural_receiver_v260.py"])
    assert not runner.is_queue_runner("python.exe", ["scripts/run_frozen_confirmation_v257.py"])
    assert not runner.is_queue_runner("python.exe", ["scripts/not_run_pure_neural_receiver_v260.py"])


def test_wait_checks_specific_queue_and_then_gpu_slot(monkeypatch):
    calls = []
    monkeypatch.setattr(runner.psutil, "process_iter", lambda fields: [])
    monkeypatch.setattr(runner, "wait_gpu_slot", lambda **kwargs: calls.append(kwargs))
    runner.wait_queue(5)
    assert 0 < calls[0]["timeout"] <= 5

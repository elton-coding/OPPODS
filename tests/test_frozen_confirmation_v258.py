import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_frozen_confirmation_v258 as runner


def valid_audit():
    return {"label": runner.LABEL, "protocol": copy.deepcopy(runner.PROTOCOL), "exact_mean_final": 68.7,
            "comparisons": {label: {
                "per_seed": [{"noise_seed": seed, "delta": {"final": .1, "efficiency": .1, "p10": -.02}}
                             for seed in runner.SEEDS],
                "paired_delta_95_percentile_interval": [.02, .15], "mean_delta": .1}
                            for label in runner.REFERENCES}}


def test_fixed_shared_window_two_comparators_no_new_training():
    cmd = runner.command()
    assert runner.OFFSET == 8000
    assert runner.SEEDS == [52701, 52702, 52703]
    assert runner.REFERENCES == ("v250_confirm_v257", "v257_confirm")
    assert cmd[cmd.index("--submission") + 1] == "artifacts/pure_neural_v258/eight/endtoend_steps72000"
    assert cmd[cmd.index("--baseline-label") + 1] == runner.REFERENCES[0]
    assert cmd[cmd.index("--control-label") + 1] == runner.REFERENCES[1]
    assert cmd[-3:] == ["52701", "52702", "52703"]
    assert "--steps" not in cmd


def test_pass_is_conditional_and_keeps_p10_tradeoff():
    result = runner.decision(valid_audit())
    assert result["supports_improvement_over_frozen_v250"]
    assert result["supports_preference_over_both_frozen_references"]
    assert result["comparisons"][runner.REFERENCES[0]]["per_seed_metric_deltas"][0]["p10"] == -.02
    for key in ("automatic_promotion", "online_confirmation", "whole_project_blindness_certified",
                "ancestor_training_provenance_certified"):
        assert result[key] is False
    assert "1812/2000" in result["caveat"]


@pytest.mark.parametrize("label", runner.REFERENCES)
@pytest.mark.parametrize("where,value", [("seed", 0), ("seed", -.1), ("interval", 0), ("interval", -.01)])
def test_both_comparisons_required_not_just_higher_average(label, where, value):
    audit = valid_audit()
    comparison = audit["comparisons"][label]
    if where == "seed":
        comparison["per_seed"][1]["delta"]["final"] = value
    else:
        comparison["paired_delta_95_percentile_interval"][0] = value
    result = runner.decision(audit)
    assert not result["supports_preference_over_both_frozen_references"]
    assert result["supports_improvement_over_frozen_v250"] == (label == runner.REFERENCES[1])


@pytest.mark.parametrize("case", ["protocol", "label", "missing_reference", "extra_reference", "nan_mean",
                                 "duplicate_seed", "short_interval", "reversed_interval", "nan_p10"])
def test_invalid_decision_rejected(case):
    audit = valid_audit()
    comparison = audit["comparisons"][runner.REFERENCES[0]]
    if case == "protocol":
        audit["protocol"]["test_offset"] = 6000
    elif case == "label":
        audit["label"] = "other"
    elif case == "missing_reference":
        audit["comparisons"].pop(runner.REFERENCES[1])
    elif case == "extra_reference":
        audit["comparisons"]["other"] = comparison
    elif case == "nan_mean":
        audit["exact_mean_final"] = float("nan")
    elif case == "duplicate_seed":
        comparison["per_seed"][1]["noise_seed"] = runner.SEEDS[0]
    elif case == "short_interval":
        comparison["paired_delta_95_percentile_interval"] = [.1]
    elif case == "reversed_interval":
        comparison["paired_delta_95_percentile_interval"] = [.2, .1]
    else:
        comparison["per_seed"][0]["delta"]["p10"] = float("nan")
    with pytest.raises(ValueError):
        runner.decision(audit)


def test_preregistration_refuses_even_one_partial_reference_output(monkeypatch, tmp_path):
    paths = [tmp_path / "partial.npz", tmp_path / "decision.json"]
    monkeypatch.setattr(runner, "reference_outputs", lambda: paths)
    runner.require_unseen_reference()
    paths[0].touch()
    with pytest.raises(FileExistsError, match="retrospectively"):
        runner.require_unseen_reference()


def cache_values(seed):
    return {"data_index": runner.deterministic_split_indices(100000, seed=1176)["test"][8000:10000].repeat(2),
            "length": np.full(4000, 1152), "score": np.full(4000, 68., dtype=np.float32),
            "snr": np.linspace(-20, 20, 4000), "noise_seed": np.array(seed),
            "split_seed": np.array(1176), "test_offset": np.array(8000)}


@pytest.mark.parametrize("case", ["valid", "wrong_ids_all_arms", "short_payload", "nan_score", "high_score",
                                 "wrong_seed", "float_metadata", "unpaired_snr", "wrong_shape"])
def test_actual_window_payload_metadata_and_pairing(monkeypatch, tmp_path, case):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    (tmp_path / "benchmarks").mkdir()
    labels = (*runner.REFERENCES, runner.LABEL)
    for label in labels:
        for seed, path in zip(runner.SEEDS, runner.score_paths(tmp_path / "benchmarks", label, runner.SEEDS, 8000), strict=True):
            values = cache_values(seed)
            if case == "wrong_ids_all_arms":
                values["data_index"] = runner.deterministic_split_indices(100000, seed=1176)["test"][6000:8000].repeat(2)
            elif label == runner.LABEL and seed == runner.SEEDS[0]:
                if case == "short_payload":
                    values["length"][0] = 1000
                elif case == "nan_score":
                    values["score"][0] = np.nan
                elif case == "high_score":
                    values["score"][0] = 101
                elif case == "wrong_seed":
                    values["noise_seed"] = np.array(52704)
                elif case == "float_metadata":
                    values["split_seed"] = np.array(1176.)
                elif case == "unpaired_snr":
                    values["snr"][1] += .01
                elif case == "wrong_shape":
                    values["length"] = values["length"][:-2]
            np.savez(path, **values)
    if case == "valid":
        result = runner.checked_caches(labels)
        assert all(abs(row["final"] - 68.) < 1e-10 for rows in result.values() for row in rows)
    else:
        with pytest.raises(ValueError):
            runner.checked_caches(labels)


def complete_audit():
    source = {"label": runner.LABEL, "directory": "model", "files": {"x": "hash"}}
    audit = {"label": runner.LABEL, "protocol": copy.deepcopy(runner.PROTOCOL), "files": source["files"],
             "training_report": {"full": True}, "exact_mean_final": 68.,
             "per_seed": [{"seed": seed, "samples": 2000, "scores": 4000, "short_outputs": 0,
                           "min_output_length": 1152, "max_output_length": 1152,
                           "efficiency": 68., "fairness": 68., "final": 68.} for seed in runner.SEEDS]}
    cached = [{"efficiency": 68., "p10": 68., "final": 68.} for _ in runner.SEEDS]
    return source, audit, cached


@pytest.mark.parametrize("case", ["valid", "wrong_weight", "partial", "wrong_raw", "wrong_mean", "nan"])
def test_full_audit_source_and_raw_statistics(monkeypatch, case):
    monkeypatch.setattr(runner, "read", lambda p: {"full": True})
    source, audit, cached = complete_audit()
    if case == "wrong_weight":
        audit["files"] = {"x": "wrong"}
    elif case == "partial":
        audit["per_seed"][0]["short_outputs"] = 1
    elif case == "wrong_raw":
        audit["per_seed"][0]["fairness"] += .01
    elif case == "wrong_mean":
        audit["exact_mean_final"] += .01
    elif case == "nan":
        audit["per_seed"][0]["final"] = float("nan")
    if case == "valid":
        runner.checked_audit(audit, source, cached)
    else:
        with pytest.raises(ValueError):
            runner.checked_audit(audit, source, cached)


def test_completed_reference_is_checked_not_retrained(monkeypatch, tmp_path):
    path = tmp_path / "decision.json"
    path.touch()
    monkeypatch.setattr(runner.reference, "RESULT_PATH", path)
    calls = []
    monkeypatch.setattr(runner, "checked_reference", lambda plan: calls.append(plan) or {"checked": True})
    assert runner.wait_reference({"frozen": True}, 1) == {"checked": True}
    assert calls == [{"frozen": True}]


def test_failed_reference_does_not_silently_unlock(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.reference, "RESULT_PATH", tmp_path / "absent.json")
    (tmp_path / "benchmarks").mkdir()
    (tmp_path / "benchmarks/v257_confirmation_failure.json").touch()
    with pytest.raises(RuntimeError, match="reference confirmation failed"):
        runner.wait_reference({}, 1)


def test_selection_cache_check_uses_actual_selection_window(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    (tmp_path / "benchmarks").mkdir()
    seeds = [22701, 22702, 22703]
    for seed, path in zip(seeds, runner.score_paths(tmp_path / "benchmarks", "selection", seeds, 2000), strict=True):
        values = cache_values(seed)
        values["test_offset"] = np.array(2000)
        values["data_index"] = runner.deterministic_split_indices(100000, seed=1176)["test"][2000:4000].repeat(2)
        np.savez(path, **values)
    assert len(runner.checked_caches(["selection"], offset=2000, seeds=seeds)["selection"]) == 3


def mock_main(monkeypatch, tmp_path, args):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    for key in ("PLAN_PATH", "RESULT_PATH", "FAILURE_PATH", "BINDING_PATH", "AUDIT_PATH"):
        monkeypatch.setattr(runner, key, tmp_path / (key + ".json"))
    monkeypatch.setattr(runner.sys, "argv", ["runner", *args])
    monkeypatch.setattr(runner, "make_plan", lambda: {"fixed": True})


def test_registration_checks_reference_absence_before_and_after_freezing(monkeypatch, tmp_path):
    mock_main(monkeypatch, tmp_path, ["--register-only"])
    calls = []
    monkeypatch.setattr(runner, "require_unseen_reference", lambda: calls.append("unseen"))
    runner.main()
    assert calls == ["unseen", "unseen"]
    assert runner.read(runner.PLAN_PATH) == {"fixed": True}
    with pytest.raises(FileExistsError):
        runner.main()


def test_reference_output_race_during_registration_refused(monkeypatch, tmp_path):
    mock_main(monkeypatch, tmp_path, ["--register-only"])
    calls = []

    def guard():
        calls.append(1)
        if len(calls) == 2:
            raise FileExistsError("reference output appeared")

    monkeypatch.setattr(runner, "require_unseen_reference", guard)
    with pytest.raises(FileExistsError):
        runner.main()
    assert not runner.PLAN_PATH.exists()


def test_existing_partial_candidate_refused_before_make_plan(monkeypatch, tmp_path):
    mock_main(monkeypatch, tmp_path, [])
    runner.AUDIT_PATH.touch()
    monkeypatch.setattr(runner, "make_plan", lambda: pytest.fail("must not proceed"))
    with pytest.raises(FileExistsError, match="no repeat"):
        runner.main()


def test_changed_registration_refused_before_lock_or_evaluation(monkeypatch, tmp_path):
    mock_main(monkeypatch, tmp_path, [])
    runner.write_new(runner.PLAN_PATH, {"fixed": False})
    with pytest.raises(RuntimeError, match="freeze differs"):
        runner.main()
    assert not (tmp_path / "artifacts/v258_confirmation.lock").exists()

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_shared_endtoend_factorial_v262 as analysis


def groups():
    base = np.arange(24).reshape(3, 4, 2) + 50.
    return {"C": base, "A": base + 1, "B": base + 2, "AB": base + 3}


def test_additive_effects_have_zero_interaction_with_paired_resampling():
    result = analysis.analyze_groups(groups(), repeats=20)
    assert result["interaction"]["mean"] == pytest.approx(0, abs=1e-12)
    assert result["interaction"]["paired_95_interval"] == pytest.approx([0, 0], abs=1e-12)
    assert result["comparisons"]["AB-C"]["mean_delta"] == pytest.approx(3)
    assert result["comparisons"]["AB-A"]["paired_delta_95_interval"] == pytest.approx([2, 2])
    assert result["comparisons"]["AB-B"]["mean_delta"] == pytest.approx(1)
    assert result["eligible_for_new_confirmation"]
    assert not result["automatic_promotion"] and not result["online_confirmation"]


def test_positive_interaction_does_not_replace_candidate_gate():
    data = groups()
    data["A"] = data["C"] - 2
    data["B"] = data["C"] - 2
    data["AB"] = data["C"] - 1
    result = analysis.analyze_groups(data, repeats=20)
    assert result["interaction"]["mean"] > 0
    assert not result["eligible_for_new_confirmation"]


def test_one_negative_noise_fails_gate_despite_positive_average():
    data = groups()
    data["AB"] = data["AB"].copy()
    data["AB"][0] = data["C"][0] - .1
    assert not analysis.analyze_groups(data, repeats=20)["eligible_for_new_confirmation"]


@pytest.mark.parametrize("case", ["missing", "nan", "range", "ue_shape", "noise_shape"])
def test_invalid_score_arrays_rejected(case):
    data = groups()
    if case == "missing":
        data.pop("A")
    elif case in ("nan", "range"):
        data["AB"] = np.full((3, 4, 2), np.nan if case == "nan" else 101.)
    else:
        shape = (3, 4, 1) if case == "ue_shape" else (2, 4, 2)
        data = {arm: np.ones(shape) * 50 for arm in data}
    with pytest.raises(ValueError):
        analysis.analyze_groups(data, repeats=2)


def save_caches(tmp_path, corruption=None):
    paths = {}
    for arm in analysis.ARMS:
        paths[arm] = []
        for seed in analysis.SEEDS:
            path = tmp_path / f"{arm}_{seed}.npz"
            values = {"score": np.full(4000, 65.), "length": np.full(4000, 1152),
                      "data_index": analysis.deterministic_split_indices(100000, seed=1176)["test"][2000:4000].repeat(2),
                      "snr": np.zeros(4000),
                      "split_seed": 1176, "test_offset": 2000, "noise_seed": seed}
            if corruption and arm == "AB" and seed == analysis.SEEDS[1]:
                if corruption in ("length", "snr", "data_index"):
                    values[corruption][0] += 1
                else:
                    values[corruption] += 1
            np.savez(path, **values)
            paths[arm].append(path)
    return paths


def test_load_exact_full_payload_paired_caches(tmp_path):
    loaded = analysis.load_groups(save_caches(tmp_path))
    assert all(value.shape == (3, 2000, 2) for value in loaded.values())


@pytest.mark.parametrize("field", ["length", "snr", "data_index", "noise_seed", "test_offset", "split_seed"])
def test_unpaired_or_wrong_protocol_caches_rejected(tmp_path, field):
    with pytest.raises(ValueError):
        analysis.load_groups(save_caches(tmp_path, field))


def test_consistent_but_wrong_window_ids_are_not_accepted(tmp_path):
    paths = save_caches(tmp_path)
    wrong_ids = analysis.deterministic_split_indices(100000, seed=1176)["test"][:2000].repeat(2)
    for group in paths.values():
        for path in group:
            with np.load(path) as cache:
                values = {name: cache[name].copy() for name in cache.files}
            values["data_index"] = wrong_ids
            np.savez(path, **values)
    with pytest.raises(ValueError, match="actual split1176"):
        analysis.load_groups(paths)

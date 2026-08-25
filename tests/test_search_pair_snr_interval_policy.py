from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.search_pair_snr_interval_policy import (
    PairedScores,
    _pair_selection,
    _seed_from_path,
    evaluate_interval,
    is_robust,
    search_intervals,
)


def _pair(seed: int, delta: np.ndarray) -> PairedScores:
    baseline = np.linspace(50.0, 70.0, delta.size)
    return PairedScores(
        seed=seed,
        baseline=baseline,
        candidate=baseline + delta,
        snr=np.array([-2.0, 1.0, -0.5, 0.5, 2.0, 3.0]),
        data_index=np.array([0, 0, 1, 1, 2, 2]),
    )


def test_seed_parser_ignores_trailing_sample_count() -> None:
    assert _seed_from_path(Path("candidate_seed2123_2000.npz")) == 2123


def test_pair_selection_routes_both_users() -> None:
    pair = _pair(1, np.zeros(6))

    any_selected, any_pairs = _pair_selection(pair, -1.0, 1.0, "any-user")
    all_selected, all_pairs = _pair_selection(pair, -1.0, 1.0, "all-user")

    assert any_pairs == 1
    assert np.array_equal(any_selected, np.array([False, False, True, True, False, False]))
    assert all_pairs == 1
    assert np.array_equal(all_selected, np.array([False, False, True, True, False, False]))


def test_any_and_all_predicates_differ_for_mixed_pair() -> None:
    pair = _pair(1, np.ones(6))

    any_result = evaluate_interval([pair], -3.0, 0.0, "any-user", blocks_per_seed=1)
    all_result = evaluate_interval([pair], -3.0, 0.0, "all-user", blocks_per_seed=1)

    assert any_result["seed_results"][0]["selected_pairs"] == 2
    assert all_result["seed_results"][0]["selected_pairs"] == 0
    assert any_result["mean_final_delta"] > 0.0
    assert all_result["mean_final_delta"] == 0.0


def test_robust_search_finds_joint_positive_interval() -> None:
    pairs = [
        _pair(1, np.array([-1.0, -1.0, 2.0, 2.0, -1.0, -1.0])),
        _pair(2, np.array([-2.0, -2.0, 3.0, 3.0, -2.0, -2.0])),
    ]

    records = search_intervals(
        pairs,
        predicate="all-user",
        grid_step=0.5,
        minimum_width=0.5,
        maximum_width=2.0,
        blocks_per_seed=1,
        robust_tolerance=1.0e-7,
    )

    assert records
    assert records[0]["low_db"] <= -0.5
    assert records[0]["high_db"] > 0.5
    assert records[0]["high_db"] - records[0]["low_db"] <= 2.0
    assert is_robust(records[0], 1.0e-7)


def test_fairness_regression_is_not_robust() -> None:
    baseline = np.full(20, 60.0)
    candidate = baseline.copy()
    candidate[:4] = 0.0
    pair = PairedScores(
        seed=1,
        baseline=baseline,
        candidate=candidate,
        snr=np.zeros(20),
        data_index=np.repeat(np.arange(10), 2),
    )

    record = evaluate_interval([pair], -0.5, 0.5, "all-user", blocks_per_seed=1)

    assert record["min_fairness_delta"] < 0.0
    assert not is_robust(record, 1.0e-7)

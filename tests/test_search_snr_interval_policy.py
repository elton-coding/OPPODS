from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.search_snr_interval_policy import (
    PairedScores,
    _seed_from_path,
    evaluate_interval,
    is_robust,
    search_intervals,
)


def test_seed_parser_ignores_trailing_sample_count() -> None:
    assert _seed_from_path(Path("candidate_seed1176_2000.npz")) == 1176


def _pair(seed: int, delta: np.ndarray) -> PairedScores:
    baseline = np.linspace(50.0, 70.0, delta.size)
    return PairedScores(
        seed=seed,
        baseline=baseline,
        candidate=baseline + delta,
        snr=np.array([-1.75, -1.25, -0.75, -0.25]),
    )


def test_evaluate_interval_routes_only_selected_scores() -> None:
    pair = _pair(1, np.array([1.0, -2.0, 3.0, -4.0]))

    result = evaluate_interval([pair], -1.0, 0.0, blocks_per_seed=1)

    seed = result["seed_results"][0]
    assert seed["selected"] == 2
    assert np.isclose(seed["score_delta_sum"], -1.0)


def test_robust_search_finds_shared_positive_interval() -> None:
    pairs = [
        _pair(1, np.array([-1.0, 1.0, 1.0, -1.0])),
        _pair(2, np.array([-2.0, 2.0, 2.0, -2.0])),
    ]

    records = search_intervals(
        pairs,
        grid_step=0.5,
        minimum_width=0.5,
        maximum_width=2.0,
        blocks_per_seed=1,
        robust_tolerance=1.0e-7,
    )

    assert records
    assert records[0]["low_db"] == -1.5
    assert records[0]["high_db"] == -0.5
    assert is_robust(records[0], 1.0e-7)


def test_fairness_regression_is_not_robust() -> None:
    baseline = np.full(20, 60.0)
    candidate = baseline.copy()
    candidate[:3] = 0.0
    pair = PairedScores(
        seed=1,
        baseline=baseline,
        candidate=candidate,
        snr=np.zeros(20),
    )

    record = evaluate_interval([pair], -0.5, 0.5, blocks_per_seed=1)

    assert record["min_fairness_delta"] < 0.0
    assert not is_robust(record, 1.0e-7)

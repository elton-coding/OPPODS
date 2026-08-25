from pathlib import Path

import numpy as np

from scripts.search_extension_policy import (
    SeedData,
    _seed_from_path,
    evaluate_policy,
    is_robust,
)


def test_seed_parser_ignores_trailing_sample_count() -> None:
    assert _seed_from_path(Path("features_seed2120_2000.npz")) == 2120


def _seed_data(reference_scores: np.ndarray) -> SeedData:
    baseline_scores = np.full(20, 50.0)
    return SeedData(
        seed=1176,
        baseline_scores=baseline_scores,
        reference_scores=reference_scores,
        positions=np.array([0], dtype=np.int64),
        fallback_scores=np.array([50.0]),
        extension_scores=np.array([60.0]),
        snr=np.array([-10.0]),
        metrics={},
    )


def test_policy_block_constraint_accepts_nonnegative_contiguous_blocks() -> None:
    data = _seed_data(np.full(20, 50.0))

    record = evaluate_policy([data], [np.array([True])], blocks_per_seed=2)

    assert record["mean_final_delta"] > 0.0
    assert record["min_block_final_delta"] == 0.0
    assert is_robust(record)


def test_policy_block_constraint_rejects_hidden_second_half_regression() -> None:
    reference_scores = np.full(20, 50.0)
    reference_scores[15] = 60.0
    data = _seed_data(reference_scores)

    record = evaluate_policy([data], [np.array([True])], blocks_per_seed=2)

    assert record["min_block_final_delta"] < 0.0
    assert not is_robust(record)

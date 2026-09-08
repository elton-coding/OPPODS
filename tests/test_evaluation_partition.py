import importlib.util
from pathlib import Path

import numpy as np
import pytest

from oppods.data import deterministic_split_indices

spec = importlib.util.spec_from_file_location(
    "partition_evaluator", Path(__file__).resolve().parents[1] / "scripts/evaluate_submission.py",
)
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


def test_fixed_partition_is_independent_of_noise_seed_and_disjoint_from_training():
    a = evaluator.evaluation_indices(10000, seed=1177, split_seed=1176, offset=200, samples=200)
    b = evaluator.evaluation_indices(10000, seed=1178, split_seed=1176, offset=200, samples=200)
    split = deterministic_split_indices(10000, seed=1176)
    assert np.array_equal(a, b)
    assert not np.intersect1d(a, split["train"]).size
    assert not np.intersect1d(a, split["validation"]).size
    assert not np.intersect1d(a, split["test"][:200]).size


def test_legacy_partition_is_reproducible_and_window_is_checked():
    legacy = evaluator.evaluation_indices(10000, seed=1177, split_seed=None, offset=0, samples=200)
    assert np.array_equal(legacy, deterministic_split_indices(10000, seed=1177)["test"][:200])
    with pytest.raises(ValueError):
        evaluator.evaluation_indices(10000, seed=1177, split_seed=1176, offset=900, samples=200)

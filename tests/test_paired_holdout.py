import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "paired_comparison", Path(__file__).resolve().parents[1] / "scripts/compare_paired_holdout.py",
)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def test_identical_candidates_have_zero_paired_uncertainty_and_mismatches_fail(tmp_path):
    path = tmp_path / "base.npz"
    values = dict(
        data_index=np.repeat(np.arange(100), 2), snr=np.linspace(-20, 19.99, 200),
        score=np.linspace(50, 99, 200), split_seed=1176, noise_seed=22701, test_offset=2000,
    )
    np.savez(path, **values)
    result = comparison.compare([path], [path], repeats=20)
    assert result["mean_delta"] == 0
    assert result["paired_delta_95_percentile_interval"] == [0, 0]
    other = tmp_path / "other.npz"
    values["noise_seed"] = 999
    np.savez(other, **values)
    with pytest.raises(ValueError, match="unpaired noise_seed"):
        comparison.compare([path], [other], repeats=20)

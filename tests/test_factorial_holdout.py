import importlib.util
from pathlib import Path

import numpy as np
import pytest


def test_factorial_reports_interaction_not_sum_of_individual_gains(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("factorial_test", scripts / "compare_factorial_holdout.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paths = {}
    for name, gain in {"C": 0, "A": 1, "B": 2, "AB": 4}.items():
        path = tmp_path / f"{name}.npz"
        np.savez(
            path, score=np.linspace(40, 90, 200) + gain,
            data_index=np.repeat(np.arange(100), 2), snr=np.linspace(-20, 19.9, 200),
            split_seed=1176, noise_seed=22701, test_offset=2000,
        )
        paths[name] = [path]
    result = module.factorial(paths, repeats=20)
    assert result["interaction_AB_minus_A_minus_B_plus_C"] == pytest.approx(1)
    assert result["interaction_paired_95_interval"] == pytest.approx([1, 1])

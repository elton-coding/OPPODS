import numpy as np
import pytest
from diagnose_train_validation_v274 import proxy_summary


def test_identical_and_reversed_tail():
    hard = np.arange(100.)
    same = proxy_summary(hard, hard)
    assert same['tail_recall'] == same['tail_precision'] == 1.
    assert same['mean_proxy_minus_hard'] == 0.
    assert proxy_summary(hard, hard[::-1])['tail_recall'] == 0.


def test_invalid_and_constant():
    with pytest.raises(ValueError):
        proxy_summary([0., 1.], [0., np.nan])
    with pytest.raises(ValueError):
        proxy_summary([0., 1.], [0.])
    assert proxy_summary(np.ones(100), np.ones(100))['pearson'] is None

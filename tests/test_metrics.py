import numpy as np
import pytest
import torch

from oppods.metrics import per_sample_score, summarize_scores


def test_official_score_extremes() -> None:
    bits = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    perfect = torch.tensor([[-1.0, 1.0, -1.0, 1.0]])
    wrong = -perfect
    one_bit = torch.tensor([[-1.0]])
    assert per_sample_score(bits, perfect).item() == 100.0
    assert per_sample_score(bits, wrong).item() == 0.0
    assert per_sample_score(bits, one_bit).item() == 62.5


def test_summary_matches_formula() -> None:
    summary = summarize_scores(np.array([40.0, 50.0, 60.0, 70.0]))
    assert summary.efficiency == 55.0
    assert summary.fairness == 43.0
    assert summary.final == pytest.approx(0.7 * 55.0 + 0.3 * 43.0)

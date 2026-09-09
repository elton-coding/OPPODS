import sys
from pathlib import Path

import numpy as np
import pytest

from oppods.data import deterministic_split_indices

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_confirmation_overlap import inspect_archive, summarize_windows


def test_channel_overlap_deduplicates_ues_and_keeps_legacy_scope(tmp_path):
    split = deterministic_split_indices(100000, seed=1176)
    ids = split["test"][4000:4004]
    path = tmp_path / "legacy.npz"
    np.savez(path, data_index=np.repeat(ids[:2], 2), score=np.array([np.nan]))
    legacy = inspect_archive(path, 100000)
    assert legacy["unique_channel_count"] == 2
    path = tmp_path / "fixed.npz"
    np.savez(path, data_index=np.repeat(ids[1:], 2), split_seed=np.array(1176))
    fixed = inspect_archive(path, 100000)
    result = summarize_windows(100000, [legacy, fixed], [4000], 4)
    window = result["windows"][0]
    assert window["all_recorded_evaluation_ids_overlap"] == 4
    assert window["explicit_split1176_recorded_ids_overlap"] == 3
    assert window["train1176_overlap"] == window["validation1176_overlap"] == 0
    assert window["current_offset2000_audit_overlap"] == 0
    assert result["blindness_certified"] is False


def test_missing_ids_are_unknown_not_evidence_of_unused_channels(tmp_path):
    path = tmp_path / "missing.npz"
    np.savez(path, score=np.array([0.5]))
    record = inspect_archive(path, 100000)
    result = summarize_windows(100000, [record], [4000], 2000)
    assert result["unresolved_archives_without_ids"] == 1
    assert result["blindness_certified"] is False


def test_invalid_channel_ids_and_window_are_rejected(tmp_path):
    path = tmp_path / "invalid.npz"
    np.savez(path, data_index=np.array([-1, 100000]))
    with pytest.raises(ValueError, match="outside"):
        inspect_archive(path, 100000)
    with pytest.raises(ValueError, match="window"):
        summarize_windows(100000, [], [9000], 2000)

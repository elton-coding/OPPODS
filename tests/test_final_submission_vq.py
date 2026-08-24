from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import torch

from scripts.build_submission import ARCHIVE_MEMBERS, package_submission

ROOT = Path(__file__).resolve().parents[1]


def _load_submission_module():
    path = ROOT / "modelSubmit" / "modelDesign.py"
    spec = importlib.util.spec_from_file_location("final_submission_model", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_threshold_tail_codebook_has_adjacent_complement_pairs() -> None:
    module = _load_submission_module()
    codebook = module._threshold_tail_codebook(torch.device("cpu"), torch.int64)
    assert codebook.shape == (module.THRESHOLD_CODEWORDS, 228)
    for index in range(0, module.THRESHOLD_CODEWORDS - 1, 2):
        assert torch.equal(codebook[index] + codebook[index + 1], torch.ones(228, dtype=torch.int64))


def test_low_snr_one_bit_guard_is_disabled_in_competition_range() -> None:
    module = _load_submission_module()
    assert module.LOW_SNR_THRESHOLD_DB == -20.0


def test_threshold_assignment_stays_in_configured_level_range() -> None:
    module = _load_submission_module()
    snr = torch.tensor([[-20.0, -19.0], [-5.0, 8.0], [10.24, 19.0]], dtype=torch.float32)
    assignment, threshold_index = module._pilot_assignment(snr)
    assert assignment.shape == (3, 2)
    assert torch.all((threshold_index >= 0) & (threshold_index < module.THRESHOLD_CODEWORDS))
    assert torch.all(assignment.sum(dim=1) == 1)


def test_final_archive_has_exact_root_members(tmp_path: Path) -> None:
    archive_path = tmp_path / "submission.zip"
    package_submission(ROOT / "modelSubmit", archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.namelist() == list(ARCHIVE_MEMBERS)
        assert archive.testzip() is None

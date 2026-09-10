import hashlib
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import verify_candidate_package_v257 as verify


def archive_fixture(tmp_path, *, extra=False, mismatch=False):
    path = tmp_path / "candidate.zip"
    files = {}
    with zipfile.ZipFile(path, "x") as archive:
        for name in verify.ARCHIVE_MEMBERS:
            content = name.encode()
            files[name.rsplit("/", 1)[1]] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
            archive.writestr(name, content + (b"wrong" if mismatch else b""))
        if extra:
            archive.writestr("unwanted.txt", "extra")
    return path, files


def test_exact_archive_layout_size_and_member_hashes(tmp_path):
    path, files = archive_fixture(tmp_path)
    members = verify.verify_archive(path, files)
    assert tuple(members) == verify.ARCHIVE_MEMBERS
    assert all(record == files[name.rsplit("/", 1)[1]] for name, record in members.items())


@pytest.mark.parametrize("case", ["extra", "mismatch"])
def test_extra_members_or_wrong_bytes_rejected(tmp_path, case):
    path, files = archive_fixture(tmp_path, **{case: True})
    with pytest.raises(ValueError):
        verify.verify_archive(path, files)


def test_corrupt_archive_rejected(tmp_path):
    path = tmp_path / "broken.zip"
    path.write_bytes(b"not a zip")
    with pytest.raises(zipfile.BadZipFile):
        verify.verify_archive(path, {})


def test_snr_contract_covers_all_routes_boundaries_and_asymmetry():
    cases = verify.snr_cases()
    assert len(cases) == 25
    assert {min(7, max(0, int((min(case)+20)//5))) for case in cases} == set(range(8))
    assert (-20., 20.) in cases and (20., -20.) in cases and (20., 20.) in cases
    assert all(-20 <= value <= 20 for case in cases for value in case)


def test_candidate_paths_never_point_to_current_release():
    assert verify.ARCHIVE != verify.ROOT / "artifacts/FATE_MIMO_submission.zip"
    assert "candidate_not_released" in verify.ARCHIVE.name
    assert verify.DIRECTORY != verify.ROOT / "modelSubmit"

import hashlib
import json
import zipfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def test_champion_config_weights_and_optional_archive_are_consistent():
    board = json.loads((ROOT / "benchmarks/leaderboard.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((ROOT / "configs/final.yaml").read_text(encoding="utf-8"))
    champion = board["champion"]
    assert config["solution_version"] == champion["version"]
    assert config["evaluation"]["mean_final"] == pytest.approx(champion["local_validation"]["mean_final"])
    assert config["evaluation"]["seeds"] == board["protocol"]["seeds"]
    assert config["submission"]["sha256"] == champion["submission_archive"]["sha256"]
    for name, expected in champion["sha256"].items():
        assert file_hash(ROOT / "modelSubmit" / name) == expected, name
    archive_path = ROOT / config["submission"]["archive"]
    # Archives are intentionally local-only and absent on fresh source clones.
    if archive_path.exists():
        assert file_hash(archive_path) == champion["submission_archive"]["sha256"]
        with zipfile.ZipFile(archive_path) as archive:
            assert len(archive.namelist()) == 4
            for name in archive.namelist():
                filename = Path(name).name
                assert hashlib.sha256(archive.read(name)).hexdigest().upper() == champion["sha256"][filename]

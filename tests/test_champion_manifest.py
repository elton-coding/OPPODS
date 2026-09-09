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
    source = json.loads((ROOT / config["training"]["source_report"]).read_text(encoding="utf-8"))
    assert source["exact_mean_final"] == pytest.approx(config["evaluation"]["mean_final"])
    assert source["training_report"]["history"][-1]["step"] == config["training"]["requested_steps"]
    assert source["training_report"]["loss_kind"] == config["training"]["loss"]
    assert {name: item["sha256"].upper() for name, item in source["files"].items()} == champion["sha256"]
    archive_path = ROOT / config["submission"]["archive"]
    # Archives are intentionally local-only and absent on fresh source clones.
    if archive_path.exists():
        assert file_hash(archive_path) == champion["submission_archive"]["sha256"]
        with zipfile.ZipFile(archive_path) as archive:
            assert len(archive.namelist()) == 4
            for name in archive.namelist():
                filename = Path(name).name
                assert hashlib.sha256(archive.read(name)).hexdigest().upper() == champion["sha256"][filename]


def test_current_champion_confirmation_preserves_scope_and_frozen_weights():
    board = json.loads((ROOT / "benchmarks/leaderboard.json").read_text(encoding="utf-8"))
    confirmation = board["champion"]["conditional_confirmation"]
    audit = json.loads((ROOT / confirmation["report"]).read_text(encoding="utf-8"))
    assert audit["exact_mean_final"] == pytest.approx(confirmation["mean_final"])
    assert audit["protocol"]["test_offset"] == confirmation["test_offset"]
    assert audit["protocol"]["noise_seeds"] == confirmation["noise_seeds"]
    assert {name: item["sha256"].upper() for name, item in audit["files"].items()} == board["champion"]["sha256"]
    paired = audit["comparisons"][confirmation["baseline_label"]]
    assert paired["paired_delta_95_percentile_interval"][0] > 0
    assert all(row["delta"]["final"] > 0 for row in paired["per_seed"])
    assert not confirmation["whole_project_blindness_certified"]
    assert not confirmation["ancestor_training_provenance_certified"]
    assert board["champion"]["online_leaderboard"] is None
    decision = json.loads((ROOT / confirmation["decision_report"]).read_text(encoding="utf-8"))
    assert decision["supports_current_cohort_promotion"]
    assert decision["candidate_exact_mean_final"] == pytest.approx(audit["exact_mean_final"])
    assert not decision["online_confirmation"]
    assert not decision["whole_project_blindness_certified"]
    assert not decision["ancestor_training_provenance_certified"]

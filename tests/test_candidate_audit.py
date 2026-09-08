import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("candidate_audit", SCRIPTS / "audit_pure_neural_candidate.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_fingerprint_binds_source_and_all_weights(tmp_path):
    for name in audit.MODEL_FILES:
        (tmp_path / name).write_bytes(name.encode())
    initial = audit.fingerprint(tmp_path)
    assert len(initial) == 4
    (tmp_path / "receiver.pth").write_bytes(b"changed")
    changed = audit.fingerprint(tmp_path)
    assert initial["receiver.pth"] != changed["receiver.pth"]
    assert initial["encoder.pth"] == changed["encoder.pth"]


def test_score_paths_bind_split_offset_and_noise(tmp_path):
    result = audit.score_paths(tmp_path, "v230_control", [22701, 22702], 2000)
    assert result[0].name == "v230_control_holdout1176_offset2000_noise22701.npz"
    assert result[1].name.endswith("noise22702.npz")
    with pytest.raises(ValueError):
        audit.score_paths(tmp_path, "../escape", [22701], 2000)

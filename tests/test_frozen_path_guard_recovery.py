import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import recover_frozen_path_guard as recovery


@pytest.mark.parametrize("key", ["research/pure_neural_v260/modelDesign.py",
    "research\\pure_neural_v260\\modelDesign.py", "RESEARCH\\PURE_NEURAL_V260\\MODELDESIGN.PY"])
def test_actual_windows_key_spellings(key):
    assert recovery.frozen_path_hash({key: "frozen"}, "research/pure_neural_v260/modelDesign.py") == "frozen"


@pytest.mark.parametrize("mapping", [{}, {"other/modelDesign.py": "wrong"},
    {"research/a.py": "one", "research\\a.py": "one"},
    {"research/a.py": "one", "RESEARCH/A.PY": "two"}])
def test_missing_or_ambiguous_keys_rejected(mapping):
    with pytest.raises(ValueError):
        recovery.frozen_path_hash(mapping, "research/a.py")


@pytest.mark.parametrize("path", ["../a.py", "D:/Source/a.py", "\\\\server\\share\\a.py"])
def test_absolute_or_traversal_target_rejected(path):
    with pytest.raises(ValueError):
        recovery.frozen_path_hash({path: "hash"}, path)


def test_fixed_lookup_still_rejects_different_design_bytes(monkeypatch):
    monkeypatch.setattr(recovery, "fingerprint", lambda path: {"modelDesign.py": {"sha256": "actual"}})
    plan = {"input_sha256": {"research\\design.py": "actual"}}
    recovery.require_design(Path("unused"), plan, "research/design.py")
    plan["input_sha256"]["research\\design.py"] = "different"
    with pytest.raises(ValueError):
        recovery.require_design(Path("unused"), plan, "research/design.py")


def test_only_known_explicit_failure_can_recover():
    failure = {"error": "KeyError('research/design.py')", "partial_outputs_preserved": True,
               "implicit_resume_allowed": False}
    recovery.require_failure(failure, "research/design.py")
    for mutated in ({**failure, "error": "CUDA out of memory"}, {**failure, "implicit_resume_allowed": True}, {}):
        with pytest.raises(ValueError):
            recovery.require_failure(mutated, "research/design.py")


@pytest.mark.parametrize("cmd", [["scripts/original.py"], ["train.py", "--output-dir", "artifacts/output"],
                                ["train.py", "--output-dir", "artifacts\\probe"]])
def test_live_original_runner_or_trainer_is_never_restarted(monkeypatch, cmd):
    process = SimpleNamespace(info={"pid": 7, "name": "python.exe", "cmdline": cmd})
    monkeypatch.setattr(recovery.psutil, "process_iter", lambda fields: [process])
    module = SimpleNamespace(__name__="original", OUTPUT="artifacts/output", PROBE="artifacts/probe")
    with pytest.raises(RuntimeError):
        recovery.require_original_stopped(module)


def test_unrelated_training_process_not_treated_as_original(monkeypatch):
    process = SimpleNamespace(info={"pid": 8, "name": "python.exe", "cmdline": ["another.py"]})
    monkeypatch.setattr(recovery.psutil, "process_iter", lambda fields: [process])
    recovery.require_original_stopped(SimpleNamespace(__name__="original", OUTPUT="out", PROBE="probe"))

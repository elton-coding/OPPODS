import copy
import json
import sys
from collections import OrderedDict
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import convert_storage_v254 as storage


def test_half_storage_preserves_input_metadata_and_integral_buffers():
    weight = torch.tensor([.1234567, -1.23456, 0.])
    state = OrderedDict(weight=weight, counter=torch.tensor(4), empty=torch.empty(0))
    state._metadata = {"": {"version": 1}}
    original = copy.deepcopy(state)
    converted, stats = storage.half_storage_state(state)
    assert converted["weight"].dtype == torch.float16
    assert converted["counter"].dtype == torch.int64 and torch.equal(converted["counter"], state["counter"])
    assert stats["floating_tensors"] == 2 and stats["unchanged_integral_tensors"] == 1
    assert stats["maximum_roundtrip_error"] > 0
    assert converted._metadata == state._metadata and converted._metadata is not state._metadata
    assert all(torch.equal(value, original[key]) for key, value in state.items())


@pytest.mark.parametrize("value", [torch.tensor([float("nan")]), torch.tensor([float("inf")]),
                                   torch.tensor([70000.]), torch.tensor([1.], dtype=torch.float16),
                                   torch.tensor([1.], dtype=torch.float64), torch.tensor([1j]), "not a tensor"])
def test_nonfinite_overflow_and_unregistered_types_rejected(value):
    with pytest.raises(ValueError):
        storage.half_storage_state({"weight": value})


def test_loading_half_checkpoint_retains_float32_math_and_exact_promoted_values(tmp_path):
    original = torch.nn.Linear(5, 3)
    converted, _ = storage.half_storage_state(original.state_dict())
    path = tmp_path / "half.pth"
    torch.save(converted, path)
    model = torch.nn.Linear(5, 3)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    assert all(p.dtype == torch.float32 for p in model.parameters())
    for name, value in model.state_dict().items():
        assert torch.equal(value, converted[name].float())
    inputs = torch.randn(4, 5)
    expected = torch.nn.functional.linear(inputs, converted["weight"].float(), converted["bias"].float())
    torch.testing.assert_close(model(inputs), expected, atol=0, rtol=0)


def test_report_is_explicit_about_zero_new_training_and_source_validation_history():
    source = {"history": [{"step": 0}, {"step": 72000}], "requested_steps": 72000, "best_step": 71000}
    before = copy.deepcopy(source)
    result = storage.transformed_training_report(source, {"source_directory": "frozen"})
    assert source == before and result["history"] == before["history"]
    result["history"][0]["step"] = 1
    assert source == before
    transform = result["post_training_transform"]
    assert transform["new_training_steps"] == 0 and transform["retrained"] is False
    assert transform["validation_metrics_are_source_float32_not_converted"] is True


def test_registered_sources_are_full_audits_and_no_recursive_conversion():
    path = ROOT / "benchmarks/v250_eight_rms_72k_audit_offset2000.json"
    audit = json.loads(path.read_text(encoding="utf-8"))
    storage.require_source_audit(audit, "eight")
    with pytest.raises(ValueError):
        storage.require_source_audit(audit, "sixteen")
    shortened = copy.deepcopy(audit)
    shortened["training_report"]["history"].pop()
    with pytest.raises(ValueError):
        storage.require_source_audit(shortened, "eight")
    audit["training_report"]["post_training_transform"] = {}
    with pytest.raises(ValueError):
        storage.require_source_audit(audit, "eight")


def test_registered_outputs_do_not_overwrite_training_sources():
    assert [arm["num_experts"] for arm in storage.ARMS.values()] == [8, 16]
    for arm in storage.ARMS.values():
        assert arm["source"] != arm["output"]
        assert arm["output"].startswith("artifacts/pure_neural_v254/storage/")

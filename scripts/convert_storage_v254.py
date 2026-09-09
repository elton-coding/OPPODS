"""V254 deterministic Tx/Rx half storage, preserving float32 inference and raw source artifacts."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import shutil
from collections import OrderedDict

import torch
from audit_pure_neural_candidate import ROOT, fingerprint
from run_pure_neural_long_budget_v250 import require_report
from run_pure_neural_lr_v237 import fingerprints
from run_pure_neural_rms_budget_v242 import require_audit
from run_pure_neural_sixteen_v254 import require_result
from train_pure_neural_snr_experts import PureNeuralLink, load_model_design

ARMS = {
    "eight": {"source_label": "v250_eight_rms_72k", "source": "artifacts/pure_neural_v250/eight/steps72000",
              "label": "v254_eight_fp16_storage", "output": "artifacts/pure_neural_v254/storage/eight_fp16",
              "num_experts": 8},
    "sixteen": {"source_label": "v254_sixteen_rms_72k", "source": "artifacts/pure_neural_v254/sixteen/steps72000",
                "label": "v254_sixteen_fp16_storage", "output": "artifacts/pure_neural_v254/storage/sixteen_fp16",
                "num_experts": 16},
}


def require_source_audit(audit, arm):
    require_audit(audit, ARMS[arm]["source_label"], 72000)
    training = audit["training_report"]
    if "post_training_transform" in training:
        raise ValueError("cannot recursively convert an already transformed source")
    (require_report(training, 72000) if arm == "eight" else require_result(training))


def half_storage_state(state):
    """Only fp32 tensors are quantized; preserve integral buffers and state-dict metadata."""
    converted = OrderedDict()
    stats = {"floating_tensors": 0, "unchanged_integral_tensors": 0,
             "source_tensor_bytes": 0, "stored_tensor_bytes": 0, "maximum_roundtrip_error": 0.}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor) or value.is_complex():
            raise ValueError(f"unsupported checkpoint value: {key}")
        stats["source_tensor_bytes"] += value.numel() * value.element_size()
        if value.is_floating_point():
            if value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError(f"expected finite float32 source tensor: {key}")
            result = value.to(dtype=torch.float16)
            if not torch.isfinite(result).all():
                raise ValueError(f"half overflow: {key}; do not clip or silently change representation")
            error = float((result.float() - value).abs().max()) if value.numel() else 0.
            stats["maximum_roundtrip_error"] = max(stats["maximum_roundtrip_error"], error)
            stats["floating_tensors"] += 1
        else:
            result = value.clone()
            stats["unchanged_integral_tensors"] += 1
        converted[key] = result
        stats["stored_tensor_bytes"] += result.numel() * result.element_size()
    if hasattr(state, "_metadata"):
        converted._metadata = copy.deepcopy(state._metadata)
    return converted, stats


def transformed_training_report(source_report, provenance):
    report = copy.deepcopy(source_report)
    report["post_training_transform"] = {
        **provenance, "kind": "float16_TxRx_storage_float32_inference", "new_training_steps": 0,
        "retrained": False, "validation_metrics_are_source_float32_not_converted": True,
        "history_scope": "original source training history, not a new converted-model training run",
    }
    return report


def cpu_outputs(directory, channel, bits, snr):
    module = load_model_design(directory / "modelDesign.py")
    link = PureNeuralLink(module).eval()
    link.load_submission(directory)
    if not all(p.dtype == torch.float32 for p in link.parameters()):
        raise ValueError("loaded inference parameters are not all float32")
    # Verify every loaded parameter/buffer exactly equals the stored value promoted to its model dtype.
    for name in ("encoder", "transmitter", "receiver"):
        state = torch.load(directory / f"{name}.pth", map_location="cpu", weights_only=True)
        actual = getattr(link, name).state_dict()
        if actual.keys() != state.keys() or not all(torch.equal(actual[key], value.to(actual[key].dtype))
                                                  for key, value in state.items()):
            raise ValueError("load_state_dict did not reproduce stored tensor values")
        del state, actual
    with torch.inference_mode():
        output = torch.cat([link(channel[i:i+1], bits[i:i+1], snr[i:i+1],
                                 generator=torch.Generator().manual_seed(26400 + i)) for i in range(len(channel))])
    if output.shape != bits.shape or output.dtype != torch.float32 or not torch.isfinite(output).all():
        raise ValueError("converted CPU interface failed")
    del link
    gc.collect()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    args = parser.parse_args()
    arm = ARMS[args.arm]
    source, output = ROOT / arm["source"], ROOT / arm["output"]
    evidence = ROOT / f"benchmarks/{arm['label']}_conversion.json"
    if output.exists() or evidence.exists():
        raise FileExistsError("conversion output exists; inspect instead of rerun/overwrite")
    audit_path = ROOT / f"benchmarks/{arm['source_label']}_audit_offset2000.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    require_source_audit(audit, args.arm)
    before = fingerprint(source)
    training_path = source / "training_report.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    if before != audit["files"] or training != audit["training_report"]:
        raise ValueError("source model or training report changed since completed audit")
    paths = [audit_path, training_path] + [ROOT / p for p in (
        "scripts/convert_storage_v254.py", "scripts/audit_pure_neural_candidate.py",
        "scripts/run_pure_neural_sixteen_v254.py", "scripts/run_pure_neural_long_budget_v250.py",
        "scripts/run_pure_neural_rms_budget_v242.py", "scripts/run_pure_neural_lr_v237.py",
        "scripts/train_pure_neural_snr_experts.py", "scripts/train_pure_neural_rms_v239.py")]
    inputs = fingerprints(paths)
    provenance = {"source_directory": arm["source"], "source_audit": str(audit_path.relative_to(ROOT)),
                  "source_training_report_sha256": inputs[str(training_path.relative_to(ROOT))],
                  "output_directory": arm["output"]}
    output.mkdir(parents=True, exist_ok=False)
    with (output / "conversion_plan.json").open("x", encoding="utf-8") as stream:
        json.dump({"arm": args.arm, "input_sha256": inputs, "source_files": before,
                   "provenance": provenance, "inference_dtype": "float32", "new_training_steps": 0}, stream, indent=2)
    torch.set_num_threads(2)
    for name in ("modelDesign.py", "encoder.pth"):
        shutil.copy2(source / name, output / name)
    rounding = {}
    for name in ("transmitter", "receiver"):
        state = torch.load(source / f"{name}.pth", map_location="cpu", weights_only=True)
        converted, rounding[name] = half_storage_state(state)
        torch.save(converted, output / f"{name}.pth")
        del state, converted
        gc.collect()
    module = load_model_design(output / "modelDesign.py")
    count = arm["num_experts"]
    if module.NUM_EXPERTS != count:
        raise ValueError("wrong expert count for registered conversion arm")
    generator = torch.Generator().manual_seed(254001)
    channel = torch.randn(count, 2, 2, 16, 144, dtype=torch.complex64, generator=generator)
    bits = torch.randint(0, 2, (count, 2, 1152), generator=generator).float()
    edges = torch.tensor(module.SNR_EXPERT_EDGES_DB)
    snr = torch.stack([(edges[:-1] + edges[1:]) / 2, torch.full((count,), 20.)], dim=1)
    snr[1::2] = snr[1::2].flip(1)
    routes = module._expert_indices(snr.amin(1)).tolist()
    if routes != list(range(count)):
        raise ValueError("CPU conversion check must cover all expert routes")
    reference, actual = cpu_outputs(source, channel, bits, snr), cpu_outputs(output, channel, bits, snr)
    files = fingerprint(output)
    if any(files[name] != before[name] for name in ("modelDesign.py", "encoder.pth")):
        raise RuntimeError("unchanged design/Encoder were modified")
    if fingerprint(source) != before or fingerprints(paths) != inputs:
        raise RuntimeError("source inputs changed during conversion")
    with (output / "training_report.json").open("x", encoding="utf-8") as stream:
        json.dump(transformed_training_report(training, provenance), stream, indent=2)
    result = {"label": arm["label"], "source_label": arm["source_label"], "source_files": before,
              "files": files, "input_sha256": inputs, "provenance": provenance, "rounding": rounding,
              "new_training_steps": 0, "inference_dtype": "float32", "encoder_copied_exactly": True,
              "cpu_check": {"routes": routes, "ue_outputs": 2 * count, "source": "synthetic seeded inputs",
                            "maximum_logit_difference": float((reference - actual).abs().max()),
                            "hard_decision_disagreements": int(((reference >= 0) != (actual >= 0)).sum()),
                            "loaded_parameters_dtype": "float32", "all_stored_values_loaded_exactly": True},
              "passed_runtime_check": True, "score_evaluated": False, "submission_size_verified": False,
              "caveat": "source history retained as provenance; converted model needs its own official-score evaluation"}
    with evidence.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

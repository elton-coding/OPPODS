"""CPU-only V253 all-route initialization proof using training channels, not scores."""
from __future__ import annotations

import json

import torch
from probe_pure_neural_routing_v252 import evaluate_initial
from run_pure_neural_lr_v237 import ROOT, fingerprints
from train_pure_neural_snr_experts import load_model_design

from oppods.data import ChannelMemmap, deterministic_split_indices

DESIGN = "research/pure_neural_v253/modelDesign.py"
REFERENCE = "research/pure_neural_v252/modelDesign.py"
PARENT = "artifacts/pure_neural_v227/joint_low"
MAPPING = [0, 0, 0, 0, 1, 1, 1, 1]
EVIDENCE = "benchmarks/v253_cpu_runtime_probe.json"


def source_paths():
    return [ROOT / p for p in (DESIGN, REFERENCE, "scripts/probe_pure_neural_pair_v253.py",
            "scripts/probe_pure_neural_routing_v252.py", "scripts/run_pure_neural_lr_v237.py",
            "scripts/train_pure_neural_snr_experts.py", "scripts/train_pure_neural_rms_v239.py",
            "src/oppods/data.py")] + [ROOT / PARENT / f"{name}.pth" for name in ("encoder", "transmitter", "receiver")]


def route_cases(module, dtype=torch.float32):
    edges = torch.tensor(module.SNR_EXPERT_EDGES_DB, dtype=dtype)
    minimum = ((edges[:-1] + edges[1:]) / 2).repeat_interleave(2)
    fraction = torch.tensor([.25, .75] * 4, dtype=dtype)
    maximum = minimum + fraction * (20. - minimum)
    snr = torch.stack([minimum, maximum], dim=1)
    snr[1::2] = snr[1::2].flip(1)
    return snr


def main():
    output = ROOT / EVIDENCE
    if output.exists():
        raise FileExistsError("V253 CPU evidence exists; never overwrite")
    torch.set_num_threads(2)
    before = fingerprints(source_paths())
    module = load_model_design(ROOT / DESIGN)
    snr = route_cases(module)
    routes = module._pair_expert_indices(snr.T).tolist()
    if routes != list(range(8)):
        raise RuntimeError("CPU probe must cover all eight pair routes")
    data = ChannelMemmap(ROOT / "ziliao/data_train/H_train.npz")
    indices = deterministic_split_indices(len(data), seed=1176)["train"][:8]
    channel = torch.from_numpy(data.read(indices))
    bits = torch.randint(0, 2, (8, 2, 1152), generator=torch.Generator().manual_seed(253)).float()
    reference = evaluate_initial(REFERENCE, MAPPING, channel, bits, snr)
    candidate = evaluate_initial(DESIGN, MAPPING, channel, bits, snr)
    diagnostics = {
        "single_sample_maximum_difference": float((candidate["singles"] - reference["singles"]).abs().max()),
        "batched_maximum_difference": float((candidate["logits"] - reference["logits"]).abs().max()),
        "batched_hard_decision_disagreements": int(((candidate["logits"] >= 0) != (reference["logits"] >= 0)).sum()),
        "loss_difference": candidate["loss"] - reference["loss"],
        "maximum_grouped_gradient_difference": {k: float((v - reference["grouped"][k]).abs().max())
                                                 for k, v in candidate["grouped"].items()},
    }
    print(json.dumps({"diagnostics": diagnostics}), flush=True)
    torch.testing.assert_close(candidate["singles"], reference["singles"], atol=0, rtol=0)
    torch.testing.assert_close(candidate["logits"], reference["logits"], atol=1e-3, rtol=1e-5)
    if diagnostics["batched_hard_decision_disagreements"]:
        raise RuntimeError("CPU batched hard decisions differ")
    if abs(diagnostics["loss_difference"]) > 1e-5:
        raise RuntimeError("CPU initial RMS loss differs")
    if not torch.equal(candidate["rng_state"], reference["rng_state"]):
        raise RuntimeError("routing changed channel RNG consumption")
    for key, value in candidate["grouped"].items():
        torch.testing.assert_close(value, reference["grouped"][key], atol=1e-5, rtol=1e-4)
    if candidate["parameters"] != 190354976 or fingerprints(source_paths()) != before:
        raise RuntimeError("parameter count or probe inputs changed")
    record = {"purpose": "CPU initialization/gradient/runtime only, not score evidence", "matched": True,
              "input_sha256": before, "reference_design": REFERENCE, "mapping": MAPPING,
              "split_seed": 1176, "train_data_indices": indices.tolist(), "snr": snr.tolist(), "routes": routes,
              "parameters": candidate["parameters"], "finite_gradient_tensors": candidate["gradient_tensors"],
              "same_channel_rng_state": True, "diagnostics": diagnostics,
              "identity_gate": "bitwise batch1; batched decisions equal; bounded rounding, loss and grouped gradients"}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()

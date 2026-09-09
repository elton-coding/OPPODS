"""V254 CPU real-parent function and full-gradient check, not score evidence."""
from __future__ import annotations

import json

import torch
from probe_pure_neural_routing_v252 import evaluate_initial
from run_pure_neural_lr_v237 import ROOT, fingerprints
from train_pure_neural_snr_experts import load_model_design

from oppods.data import ChannelMemmap, deterministic_split_indices

DESIGN = "research/pure_neural_v254/modelDesign.py"
PARENT = "artifacts/pure_neural_v227/joint_low"
MAPPING = [0] * 4 + [1] * 12
EVIDENCE = "benchmarks/v254_cpu_runtime_probe.json"


def source_paths():
    return [ROOT / name for name in (
        DESIGN, "research/pure_neural_v230/modelDesign.py", "scripts/probe_pure_neural_sixteen_v254.py",
        "scripts/probe_pure_neural_routing_v252.py", "scripts/train_pure_neural_snr_experts.py",
        "scripts/train_pure_neural_rms_v239.py", "src/oppods/data.py")] + [
        ROOT / PARENT / name for name in ("encoder.pth", "transmitter.pth", "receiver.pth")]


def main():
    output = ROOT / EVIDENCE
    if output.exists():
        raise FileExistsError("V254 CPU evidence exists; inspect rather than overwrite")
    torch.set_num_threads(2)
    before = fingerprints(source_paths())
    module = load_model_design(ROOT / DESIGN)
    edges = torch.tensor(module.SNR_EXPERT_EDGES_DB)
    midpoints = (edges[:-1] + edges[1:]) / 2
    snr = torch.stack([midpoints, torch.full_like(midpoints, 20.)], dim=1)
    snr[1::2] = snr[1::2].flip(1)
    routes = module._expert_indices(snr.amin(dim=1)).tolist()
    if routes != list(range(16)):
        raise ValueError("all sixteen routes must be represented")
    data = ChannelMemmap(ROOT / "ziliao/data_train/H_train.npz")
    indices = deterministic_split_indices(len(data), seed=1176)["train"][:16]
    channel = torch.from_numpy(data.read(indices))
    bits = torch.randint(0, 2, (16, 2, 1152), generator=torch.Generator().manual_seed(254)).float()
    control = evaluate_initial("research/pure_neural_v230/modelDesign.py", [0, 0, 1, 1, 1, 1, 1, 1],
                               channel, bits, snr)
    candidate = evaluate_initial(DESIGN, MAPPING, channel, bits, snr)
    diagnostics = {
        "single_sample_maximum_difference": float((candidate["singles"] - control["singles"]).abs().max()),
        "batched_maximum_difference": float((candidate["logits"] - control["logits"]).abs().max()),
        "batched_hard_decision_disagreements": int(((candidate["logits"] >= 0) != (control["logits"] >= 0)).sum()),
        "loss_difference": candidate["loss"] - control["loss"],
        "maximum_grouped_gradient_difference": {key: float((value - control["grouped"][key]).abs().max())
                                                 for key, value in candidate["grouped"].items()},
    }
    print(json.dumps({"diagnostics": diagnostics}), flush=True)
    torch.testing.assert_close(candidate["singles"], control["singles"], atol=0, rtol=0)
    torch.testing.assert_close(candidate["logits"], control["logits"], atol=1e-3, rtol=1e-5)
    if diagnostics["batched_hard_decision_disagreements"] or abs(diagnostics["loss_difference"]) > 1e-5:
        raise RuntimeError("batch decisions or loss failed preregistered initialization gate")
    if not torch.equal(candidate["rng_state"], control["rng_state"]):
        raise RuntimeError("simulated channel RNG consumption changed")
    for key, value in candidate["grouped"].items():
        torch.testing.assert_close(value, control["grouped"][key], atol=1e-5, rtol=1e-4)
    if before != fingerprints(source_paths()):
        raise RuntimeError("CPU proof input changed")
    if candidate["parameters"] != 379906560 or candidate["gradient_tensors"] != 2592:
        raise ValueError("unexpected parameter count or incomplete sixteen-expert gradients")
    result = {"purpose": "CPU initialization/full-gradient runtime only; not performance evaluation",
              "matched": True, "input_sha256": before, "split_seed": 1176,
              "train_data_indices": indices.tolist(), "snr": snr.tolist(), "routes": routes, "mapping": MAPPING,
              "parameters": candidate["parameters"], "finite_gradient_tensors": candidate["gradient_tensors"],
              "same_channel_rng_state": True, "rounding_diagnostics": diagnostics,
              "identity_gate": "bitwise batch1 logits; batched decisions equal; bounded rounding/loss/head gradients"}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

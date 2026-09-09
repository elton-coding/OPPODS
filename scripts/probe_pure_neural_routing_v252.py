"""CPU-only real-parent function/gradient check; never a performance evaluation."""
from __future__ import annotations

import gc
import json

import torch
from run_pure_neural_lr_v237 import ROOT, fingerprints
from train_pure_neural_rms_v239 import rms_score_loss
from train_pure_neural_snr_experts import PureNeuralLink, load_model_design

from oppods.data import ChannelMemmap, deterministic_split_indices

DESIGN = "research/pure_neural_v252/modelDesign.py"
PARENT = "artifacts/pure_neural_v227/joint_low"
MAPPING = [0, 0, 0, 0, 1, 1, 1, 1]
EVIDENCE = "benchmarks/v252_cpu_runtime_probe.json"


def source_paths():
    return [ROOT / name for name in (DESIGN, "research/pure_neural_v230/modelDesign.py",
            "scripts/probe_pure_neural_routing_v252.py", "scripts/train_pure_neural_snr_experts.py",
            "scripts/train_pure_neural_rms_v239.py", "src/oppods/data.py")] + [
            ROOT / PARENT / name for name in ("encoder.pth", "transmitter.pth", "receiver.pth")]


def evaluate_initial(design, mapping, channel, bits, snr):
    torch.manual_seed(15240)
    module = load_model_design(ROOT / design)
    link = PureNeuralLink(module)
    link.initialize_from_expert_bank(ROOT / PARENT, mapping)
    link.train()
    # Full forward/backward through the real Tx/Rx, exactly the formal trainable set.
    link.encoder.requires_grad_(False)
    with torch.no_grad():
        singles = torch.cat([link(channel[i:i+1], bits[i:i+1], snr[i:i+1],
                                  generator=torch.Generator().manual_seed(16252 + i))
                             for i in range(len(channel))])
    generator = torch.Generator().manual_seed(15252)
    logits = link(channel, bits, snr, generator=generator)
    loss = rms_score_loss(logits, bits, loss_kind="rms_score", margin=1., tail_weight=.3,
                          tail_fraction=.1, score_temperature=.5, quantile_bandwidth=.025,
                          score_bce_weight=.05, score_fairness_weight=.3)
    loss.backward()
    gradients = [p.grad for p in link.parameters() if p.grad is not None]
    if not gradients or not torch.isfinite(loss) or not all(torch.isfinite(g).all() for g in gradients):
        raise RuntimeError("nonfinite or missing CPU training gradients")
    # Aggregate output-head gradients by source expert: changed clone allocation
    # must preserve the initial shared-parent directional derivative.
    grouped = {}
    for component, key in (("transmitter", "_out.weight"), ("receiver", "_fc_out.weight")):
        for source in (0, 1):
            selected = [dict(expert.named_parameters())[key].grad for expert, parent in
                        zip(getattr(link, component).experts, mapping, strict=True) if parent == source]
            selected = [value for value in selected if value is not None]
            grouped[f"{component}_{source}"] = torch.stack(selected).sum(0).detach().clone()
    result = {"logits": logits.detach().clone(), "singles": singles, "loss": float(loss.detach()), "grouped": grouped,
              "gradient_tensors": len(gradients), "parameters": sum(p.numel() for p in link.parameters()),
              "rng_state": generator.get_state().clone()}
    del link, logits, loss, gradients
    gc.collect()
    return result


def main():
    output = ROOT / EVIDENCE
    if output.exists():
        raise FileExistsError("CPU evidence exists; inspect it rather than overwrite")
    torch.set_num_threads(2)
    before = fingerprints(source_paths())
    module = load_model_design(ROOT / DESIGN)
    boundaries = torch.tensor(module.SNR_EXPERT_EDGES_DB)
    midpoints = (boundaries[:-1] + boundaries[1:]) / 2
    snr = torch.stack([midpoints, torch.full_like(midpoints, 20.)], dim=1)
    snr[1::2] = snr[1::2].flip(1)
    routes = module._expert_indices(snr.amin(dim=1)).tolist()
    if routes != list(range(8)):
        raise RuntimeError("CPU cases must cover all eight candidate routes")
    data = ChannelMemmap(ROOT / "ziliao/data_train/H_train.npz")
    indices = deterministic_split_indices(len(data), seed=1176)["train"][:8]
    channel = torch.from_numpy(data.read(indices))
    bits = torch.randint(0, 2, (8, 2, 1152), generator=torch.Generator().manual_seed(252)).float()
    control = evaluate_initial("research/pure_neural_v230/modelDesign.py", [0, 0, 1, 1, 1, 1, 1, 1], channel, bits, snr)
    candidate = evaluate_initial(DESIGN, MAPPING, channel, bits, snr)
    diagnostics = {"single_sample_maximum_difference": float((candidate["singles"] - control["singles"]).abs().max()),
                   "batched_maximum_difference": float((candidate["logits"] - control["logits"]).abs().max()),
                   "batched_hard_decision_disagreements": int(((candidate["logits"] >= 0) != (control["logits"] >= 0)).sum()),
                   "maximum_grouped_gradient_difference": {key: float((value - control["grouped"][key]).abs().max())
                                                            for key, value in candidate["grouped"].items()}}
    print(json.dumps({"diagnostics": diagnostics}), flush=True)
    torch.testing.assert_close(candidate["singles"], control["singles"], atol=0, rtol=0)
    # Repartitioning changes GEMM batch sizes, while single-sample execution above
    # must be bitwise equal. Bound batched rounding and require unchanged decisions.
    torch.testing.assert_close(candidate["logits"], control["logits"], atol=1e-3, rtol=1e-5)
    if diagnostics["batched_hard_decision_disagreements"]:
        raise RuntimeError("batched initialization changed hard decisions in CPU probe")
    if not torch.equal(candidate["rng_state"], control["rng_state"]):
        raise RuntimeError("routing changed simulated channel RNG consumption")
    grad_differences = {}
    for key, value in candidate["grouped"].items():
        torch.testing.assert_close(value, control["grouped"][key], atol=1e-5, rtol=1e-4)
        grad_differences[key] = float((value - control["grouped"][key]).abs().max())
    if before != fingerprints(source_paths()):
        raise RuntimeError("probe inputs changed")
    record = {"purpose": "CPU initialization and gradient/runtime only, not score evidence", "matched": True,
              "input_sha256": before, "split_seed": 1176, "train_data_indices": indices.tolist(),
              "snr": snr.tolist(), "routes": routes, "mapping": MAPPING, "parameters": candidate["parameters"],
              "maximum_logit_difference": float((candidate["logits"] - control["logits"]).abs().max()),
              "loss_difference": candidate["loss"] - control["loss"],
              "maximum_grouped_head_gradient_difference": grad_differences,
              "finite_gradient_tensors": candidate["gradient_tensors"], "same_channel_rng_state": True,
              "rounding_diagnostics": diagnostics,
              "earlier_batched_atol1e-4_check_failed": True,
              "identity_gate": "bitwise batch1 logits; batched hard decisions equal; bounded rounding and grouped gradients"}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()

"""Full batch100/all16-routes/two-Adam-step GPU allocation probe; no score claim."""
from __future__ import annotations

import json
import os

import torch
from probe_pure_neural_sixteen_v254 import DESIGN, EVIDENCE, MAPPING, PARENT, source_paths
from run_pure_neural_lr_v237 import ROOT, fingerprints
from run_pure_neural_pair_v253 import check_training_slot
from train_pure_neural_rms_v239 import rms_score_loss
from train_pure_neural_snr_experts import PureNeuralLink, load_model_design, select_trainable_parameters

from oppods.data import ChannelMemmap, deterministic_split_indices

OUTPUT = ROOT / "benchmarks/v254_full_state_gpu_probe.json"
PLAN = ROOT / "artifacts/resource_probe/v254/full_state/execution_plan.json"
MEMORY_FRACTION = .5


def forced_snr(device):
    routes = torch.arange(100, device=device) % 16
    minimum = -18.75 + 2.5 * routes.float()
    snr = torch.stack([minimum, torch.full_like(minimum, 20.)], dim=1)
    snr[1::2] = snr[1::2].flip(1)
    return snr


def main():
    if OUTPUT.exists() or PLAN.exists():
        raise FileExistsError("V254 full-state GPU probe already attempted; inspect, do not overwrite")
    cpu = json.loads((ROOT / EVIDENCE).read_text(encoding="utf-8"))
    if cpu.get("matched") is not True or fingerprints(source_paths()) != cpu["input_sha256"]:
        raise ValueError("CPU proof missing or stale")
    active = check_training_slot()
    paths = source_paths() + [ROOT / name for name in (
        EVIDENCE, "scripts/probe_pure_neural_sixteen_memory_v254.py",
        "scripts/run_pure_neural_pair_v253.py", "ziliao/data_train/H_train.npz")]
    before = fingerprints(paths)
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    with PLAN.open("x", encoding="utf-8") as stream:
        json.dump({"input_sha256": before, "other_registered_training_pids": active, "pid": os.getpid(),
                   "batch_size": 100, "adam_steps": 2, "all_routes_forced": 16,
                   "gpu_memory_fraction": MEMORY_FRACTION, "formal_training": False}, stream, indent=2)
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION, 0)
    torch.cuda.reset_peak_memory_stats(0)
    torch.manual_seed(15240)
    torch.set_num_threads(2)
    module = load_model_design(ROOT / DESIGN)
    link = PureNeuralLink(module)
    link.initialize_from_expert_bank(ROOT / PARENT, MAPPING)
    link.to(device).train()
    parameters = select_trainable_parameters(link, ["transmitter", "receiver"], None)
    optimizer = torch.optim.Adam(parameters, lr=1e-5)
    generator = torch.Generator(device=device).manual_seed(15254)
    data = ChannelMemmap(ROOT / "ziliao/data_train/H_train.npz")
    indices = deterministic_split_indices(len(data), seed=1176)["train"][:100]
    channel = torch.from_numpy(data.read(indices)).to(device)
    snr = forced_snr(device)
    counts = torch.bincount(module._expert_indices(snr.amin(dim=1)), minlength=16).cpu().tolist()
    if counts != [7] * 4 + [6] * 12:
        raise ValueError("GPU probe does not cover all sixteen routes")
    records = []
    for step in (1, 2):
        bits = torch.randint(0, 2, (100, 2, 1152), device=device, dtype=torch.float32, generator=generator)
        logits = link(channel, bits, snr, generator=generator)
        loss = rms_score_loss(logits, bits, loss_kind="rms_score", margin=.5, tail_weight=0.,
                              tail_fraction=.1, score_temperature=.5, quantile_bandwidth=.025,
                              score_bce_weight=.05, score_fairness_weight=.3)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [p.grad for p in parameters if p.grad is not None]
        if len(gradients) != 2592 or not all(torch.isfinite(g).all() for g in gradients):
            raise RuntimeError("missing or nonfinite gradient: all sixteen experts must participate")
        norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.)
        optimizer.step()
        torch.cuda.synchronize()
        if (not torch.isfinite(loss) or not torch.isfinite(norm) or len(optimizer.state) != len(parameters)
                or not all(torch.isfinite(p).all() for p in parameters)):
            raise RuntimeError("nonfinite update or incomplete Adam state")
        records.append({"step": step, "loss": float(loss.detach()), "gradient_norm_before_clip": float(norm),
                        "gradient_tensors": len(gradients), "adam_parameter_states": len(optimizer.state)})
        del logits, loss, gradients
    if len(parameters) != 2592 or sum(p.numel() for p in parameters) != 379103168:
        raise ValueError("unexpected formal trainable set")
    state_bytes = sum(value.numel() * value.element_size() for state in optimizer.state.values()
                      for value in state.values() if isinstance(value, torch.Tensor))
    peak_allocated, peak_reserved = torch.cuda.max_memory_allocated(0), torch.cuda.max_memory_reserved(0)
    if peak_allocated <= 0 or fingerprints(paths) != before:
        raise RuntimeError("GPU allocation missing or frozen input changed")
    result = {"purpose": "all-sixteen-route allocation and optimizer runtime; not formal training or performance",
              "input_sha256": before, "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "gpu_memory_fraction": MEMORY_FRACTION, "batch_size": 100, "route_counts": counts,
              "steps": records, "trainable_parameters": 379103168, "optimizer_state_bytes": state_bytes,
              "gpu_peak_allocated_bytes": peak_allocated, "gpu_peak_reserved_bytes": peak_reserved,
              "train_data_indices": indices.tolist(), "weights_saved": False, "passed": True}
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

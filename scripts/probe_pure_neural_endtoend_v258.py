"""Real-parent CPU identity/gradient proof and separately gated CUDA state probe."""
from __future__ import annotations

import argparse
import json
import time

import torch
from run_pure_neural_lr_v237 import ROOT, fingerprints
from run_pure_neural_pair_v253 import check_training_slot
from train_pure_neural_rms_v239 import rms_score_loss
from train_pure_neural_snr_experts import PureNeuralLink, load_model_design, select_trainable_parameters

from oppods.data import ChannelMemmap, deterministic_split_indices

DESIGN = "research/pure_neural_v230/modelDesign.py"
PARENT = "artifacts/pure_neural_v227/joint_low"
MAPPING = [0, 0, 1, 1, 1, 1, 1, 1]
CPU_PROOF = "benchmarks/v258_cpu_endtoend_probe.json"
GPU_PROOF = "benchmarks/v258_full_state_gpu_probe.json"
COMPONENTS = ["encoder", "transmitter", "receiver"]


def source_paths():
    names = [DESIGN, "scripts/probe_pure_neural_endtoend_v258.py",
             "scripts/train_pure_neural_snr_experts.py", "scripts/train_pure_neural_rms_v239.py",
             "scripts/run_pure_neural_lr_v237.py", "scripts/run_pure_neural_pair_v253.py",
             "src/oppods/data.py", "ziliao/data_train/H_train.npz"]
    return [ROOT / n for n in names] + [ROOT / PARENT / f"{n}.pth" for n in COMPONENTS]


def objective(logits, bits):
    return rms_score_loss(logits, bits, loss_kind="rms_score", margin=.5, tail_weight=0.,
                          tail_fraction=.1, score_temperature=.5, quantile_bandwidth=.025,
                          score_bce_weight=.05, score_fairness_weight=.3, valid_lengths=None)


def gradient_record(link):
    result = {}
    for name in COMPONENTS:
        parameters = list(getattr(link, name).parameters())
        if not parameters or any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError(f"missing/nonfinite {name} gradient")
        magnitude = sum(float(p.grad.abs().sum()) for p in parameters)
        if magnitude <= 0:
            raise ValueError(f"zero {name} gradient")
        result[name] = {"parameters": sum(p.numel() for p in parameters),
                        "gradient_tensors": len(parameters), "gradient_l1": magnitude}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    cuda = args.device == "cuda"
    path = ROOT / (GPU_PROOF if cuda else CPU_PROOF)
    if path.exists():
        raise FileExistsError("probe evidence exists; no overwrite/restart")
    torch.set_num_threads(2)
    paths = source_paths()
    if cuda:
        check_training_slot()
        cpu = json.loads((ROOT / CPU_PROOF).read_text())
        if cpu.get("passed") is not True or cpu.get("input_sha256") != fingerprints(paths):
            raise RuntimeError("CPU identity/gradient proof missing or changed")
        paths.append(ROOT / CPU_PROOF)
        torch.cuda.set_per_process_memory_fraction(.4)
        torch.cuda.reset_peak_memory_stats()
    before, started = fingerprints(paths), time.perf_counter()
    device = torch.device(args.device)
    batch = 100 if cuda else 8
    data = ChannelMemmap(ROOT / "ziliao/data_train/H_train.npz")
    indices = deterministic_split_indices(len(data), seed=1176)["train"][:batch]
    channel = torch.from_numpy(data.read(indices)).to(device)
    bits = torch.randint(0, 2, (batch, 2, 1152), generator=torch.Generator().manual_seed(15258)).float().to(device)
    routes = torch.arange(batch) % 8
    snr = torch.stack([-17.5 + 5. * routes, torch.full((batch,), 20.)], dim=1).to(device)
    snr[1::2] = snr[1::2].flip(1)
    torch.manual_seed(15240)
    module = load_model_design(ROOT / DESIGN)
    if module._expert_indices(snr.amin(1)).tolist() != routes.tolist():
        raise ValueError("probe must cover all eight routes")
    link = PureNeuralLink(module)
    link.initialize_from_expert_bank(ROOT / PARENT, MAPPING)
    link = link.to(device).train()
    construction_rng = torch.get_rng_state().clone()
    identity = None
    if not cuda:
        frozen = select_trainable_parameters(link, ["transmitter", "receiver"])
        assert sum(p.numel() for p in frozen) == 189551584
        generator = torch.Generator(device=device).manual_seed(16258)
        control_logits = link(channel, bits, snr, generator=generator)
        control_loss = objective(control_logits, bits)
        control_loss.backward()
        control_rng = generator.get_state().clone()
        assert all(p.grad is None for p in link.encoder.parameters())
        gradients = {name: p.grad.detach().clone() for name, p in link.named_parameters() if p.requires_grad}
        control_logits = control_logits.detach()
        control_value = float(control_loss.detach())
        del control_loss
        link.zero_grad(set_to_none=True)
    parameters = select_trainable_parameters(link, COMPONENTS)
    assert sum(p.numel() for p in parameters) == 190354976
    assert len(parameters) == len({id(p) for p in parameters})
    assert all(p.requires_grad for p in link.parameters())
    encoder_before = {name: p.detach().clone() for name, p in link.encoder.named_parameters()}
    optimizer = torch.optim.Adam(parameters, lr=1e-5)
    rows = []
    for step in (1, 2):
        optimizer.zero_grad(set_to_none=True)
        generator = torch.Generator(device=device).manual_seed(16257 + step)
        logits = link(channel, bits, snr, generator=generator)
        loss = objective(logits, bits)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite objective")
        loss.backward()
        component_gradients = gradient_record(link)
        if step == 1 and not cuda:
            torch.testing.assert_close(logits.detach(), control_logits, atol=0, rtol=0)
            assert float(loss.detach()) == control_value
            assert torch.equal(generator.get_state(), control_rng)
            assert torch.equal(torch.get_rng_state(), construction_rng)
            difference = 0.
            for name, p in link.named_parameters():
                if name.startswith("encoder."):
                    continue
                torch.testing.assert_close(p.grad, gradients[name], atol=1e-5, rtol=1e-4)
                difference = max(difference, float((p.grad - gradients[name]).abs().max()))
            identity = {"logit_maximum_difference": 0., "loss_difference": 0.,
                        "rng_equal": True, "frozen_encoder_has_no_gradients": True,
                        "transceiver_gradient_maximum_difference_before_clip": difference}
            del gradients, control_logits
        norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.)
        optimizer.step()
        if len(optimizer.state) != len(parameters):
            raise ValueError("incomplete all-route Adam state")
        for p in parameters:
            if not torch.isfinite(p).all() or optimizer.state[p]["step"].item() != step:
                raise ValueError("nonfinite parameters or wrong Adam update count")
            for value in optimizer.state[p].values():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    raise ValueError("nonfinite Adam state")
        delta = max(float((p.detach() - encoder_before[name]).abs().max())
                    for name, p in link.encoder.named_parameters())
        if delta <= 0:
            raise ValueError("Encoder did not actually update")
        rows.append({"step": step, "loss": float(loss.detach()), "gradient_norm_before_clip": float(norm),
                     "component_gradients": component_gradients, "adam_parameter_states": len(optimizer.state),
                     "encoder_maximum_parameter_change_from_initial": delta})
    if cuda:
        torch.cuda.synchronize()
    if before != fingerprints(paths):
        raise RuntimeError("probe inputs changed")
    result = {"purpose": "runtime/gradient evidence, not a competition score", "device": args.device,
              "input_sha256": before, "batch_size": batch, "route_counts": torch.bincount(routes, minlength=8).tolist(),
              "trainable_parameters": 190354976, "trainable_parameter_tensors": len(parameters),
              "identity": identity, "steps": rows, "weights_saved": False, "passed": True,
              "elapsed_seconds": time.perf_counter() - started,
              "gpu_memory_fraction": .4 if cuda else None,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else 0,
              "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved() if cuda else 0}
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "input_sha256"}), flush=True)


if __name__ == "__main__":
    main()

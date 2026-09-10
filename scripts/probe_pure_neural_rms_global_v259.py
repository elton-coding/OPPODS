"""Real-parent RMS global-loss checkpoint proof; separately gated CUDA batch400 runtime."""
from __future__ import annotations

import argparse
import json
import time

import torch
from run_pure_neural_lr_v237 import ROOT, fingerprints
from run_pure_neural_pair_v253 import check_training_slot
from train_pure_neural_global_batch_v246 import chunked_forward
from train_pure_neural_rms_v239 import rms_score_loss
from train_pure_neural_snr_experts import PureNeuralLink, load_model_design, select_trainable_parameters

from oppods.data import ChannelMemmap, deterministic_split_indices

DESIGN = "research/pure_neural_v230/modelDesign.py"
PARENT = "artifacts/pure_neural_v227/joint_low"
MAPPING = [0, 0, 1, 1, 1, 1, 1, 1]
CPU_PROOF = "benchmarks/v259_cpu_global_rms_probe.json"
GPU_PROOF = "benchmarks/v259_full_state_gpu_probe.json"


def source_paths():
    names = [DESIGN, "scripts/probe_pure_neural_rms_global_v259.py",
             "scripts/train_pure_neural_rms_global_v259.py", "scripts/train_pure_neural_global_batch_v246.py",
             "scripts/train_pure_neural_snr_experts.py", "scripts/train_pure_neural_rms_v239.py",
             "scripts/run_pure_neural_lr_v237.py", "scripts/run_pure_neural_pair_v253.py",
             "src/oppods/data.py", "ziliao/data_train/H_train.npz"]
    return [ROOT / n for n in names] + [ROOT / PARENT / f"{n}.pth"
                                      for n in ("encoder", "transmitter", "receiver")]


def objective(logits, bits):
    return rms_score_loss(logits, bits, loss_kind="rms_score", margin=.5, tail_weight=0.,
                          tail_fraction=.1, score_temperature=.5, quantile_bandwidth=.025,
                          score_bce_weight=.05, score_fairness_weight=.3, valid_lengths=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    cuda = args.device == "cuda"
    output = ROOT / (GPU_PROOF if cuda else CPU_PROOF)
    if output.exists():
        raise FileExistsError("proof already exists; no overwrite")
    torch.set_num_threads(2)
    paths = source_paths()
    if cuda:
        check_training_slot()
        cpu = json.loads((ROOT / CPU_PROOF).read_text())
        if cpu.get("passed") is not True or cpu.get("input_sha256") != fingerprints(paths):
            raise RuntimeError("CPU proof missing or stale")
        paths.append(ROOT / CPU_PROOF)
        torch.cuda.set_per_process_memory_fraction(.4)
        torch.cuda.reset_peak_memory_stats()
    before, started = fingerprints(paths), time.perf_counter()
    device = torch.device(args.device)
    batch, micro = (400, 100) if cuda else (16, 8)
    data = ChannelMemmap(ROOT / "ziliao/data_train/H_train.npz")
    indices = deterministic_split_indices(len(data), seed=1176)["train"][:batch]
    channel = torch.from_numpy(data.read(indices)).to(device)
    bits = torch.randint(0, 2, (batch, 2, 1152), generator=torch.Generator().manual_seed(15259)).float().to(device)
    routes = torch.arange(batch) % 8
    snr = torch.stack([-17.5 + 5. * routes, torch.full((batch,), 20.)], dim=1).to(device)
    snr[1::2] = snr[1::2].flip(1)
    torch.manual_seed(15240)
    module = load_model_design(ROOT / DESIGN)
    assert module._expert_indices(snr.amin(1)).tolist() == routes.tolist()
    link = PureNeuralLink(module)
    link.initialize_from_expert_bank(ROOT / PARENT, MAPPING)
    link = link.to(device).train()
    parameters = select_trainable_parameters(link, ["transmitter", "receiver"])
    assert len(parameters) == len({id(p) for p in parameters}) == 1296
    assert sum(p.numel() for p in parameters) == 189551584
    encoder_before = {n: p.detach().clone() for n, p in link.encoder.named_parameters()}
    optimizer = torch.optim.Adam(parameters, lr=1e-5)
    reference, identity, rows = None, None, []
    for step in ([1, 2] if cuda else [0, 1, 2]):
        optimizer.zero_grad(set_to_none=True)
        generator = torch.Generator(device=device).manual_seed(16259 if step <= 1 else 16260)
        logits = chunked_forward(link, channel, bits, snr, generator=generator,
                                 microbatch=micro, recompute=step != 0)
        rng_after_forward = generator.get_state().clone()
        # One objective over all channels/UEs; never average microbatch quantiles.
        loss = objective(logits, bits)
        loss.backward()
        if not torch.equal(generator.get_state(), rng_after_forward):
            raise RuntimeError("checkpoint backward changed future noise RNG")
        if (not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters)
                or any(p.requires_grad or p.grad is not None for p in link.encoder.parameters())):
            raise RuntimeError("nonfinite/missing gradients or unfrozen Encoder")
        if step == 0:
            reference = (logits.detach().clone(), float(loss.detach()), rng_after_forward,
                         [p.grad.detach().clone() for p in parameters])
            del logits, loss
            continue
        if step == 1 and not cuda:
            torch.testing.assert_close(logits.detach(), reference[0], atol=0, rtol=0)
            assert float(loss.detach()) == reference[1]
            assert torch.equal(rng_after_forward, reference[2])
            difference = 0.
            for p, expected in zip(parameters, reference[3], strict=True):
                torch.testing.assert_close(p.grad, expected, atol=1e-7, rtol=1e-5)
                difference = max(difference, float((p.grad - expected).abs().max()))
            identity = {"logit_maximum_difference": 0., "loss_difference": 0.,
                        "maximum_gradient_difference": difference, "forward_noise_rng_equal": True,
                        "backward_preserves_noise_rng": True, "comparison": "same chunked forwards, recompute false/true"}
            del reference
        norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.)
        optimizer.step()
        if len(optimizer.state) != 1296:
            raise RuntimeError("incomplete all-route Adam state")
        for p in parameters:
            if not torch.isfinite(p).all() or optimizer.state[p]["step"].item() != step:
                raise RuntimeError("nonfinite weights or wrong Adam updates")
            if any(torch.is_tensor(v) and not torch.isfinite(v).all() for v in optimizer.state[p].values()):
                raise RuntimeError("nonfinite Adam state")
        for name, p in link.encoder.named_parameters():
            assert torch.equal(p, encoder_before[name])
        rows.append({"step": step, "loss": float(loss.detach()), "gradient_norm_before_clip": float(norm),
                     "gradient_tensors": len(parameters), "adam_parameter_states": len(optimizer.state),
                     "global_loss_ue_count": 2 * batch, "backward_preserves_noise_rng": True})
    if cuda:
        torch.cuda.synchronize()
    if before != fingerprints(paths):
        raise RuntimeError("probe inputs changed")
    result = {"purpose": "RMS global-loss recomputation/runtime proof, not a score", "device": args.device,
              "input_sha256": before, "batch_size": batch, "microbatch_size": micro,
              "route_counts": torch.bincount(routes, minlength=8).tolist(), "trainable_parameters": 189551584,
              "identity": identity, "steps": rows, "encoder_unchanged": True, "weights_saved": False, "passed": True,
              "elapsed_seconds": time.perf_counter() - started, "gpu_memory_fraction": .4 if cuda else None,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else 0,
              "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved() if cuda else 0}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "input_sha256"}), flush=True)


if __name__ == "__main__":
    main()

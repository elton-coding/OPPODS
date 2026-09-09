"""CPU full72k Adam-hook equivalence; optimizer implementation proof, not model score."""
from __future__ import annotations

import json
import math
import time

import torch
from run_pure_neural_lr_v237 import ROOT, fingerprints
from train_pure_neural_cosine_v255 import RateRecorder, require_schedule

EVIDENCE = "benchmarks/v255_cosine_cpu_proof.json"


def source_paths():
    return [ROOT / p for p in ("scripts/probe_pure_neural_cosine_v255.py",
            "scripts/train_pure_neural_cosine_v255.py", "scripts/train_pure_neural_rms_v239.py",
            "scripts/train_pure_neural_snr_experts.py", "scripts/run_pure_neural_lr_v237.py")]


def main():
    evidence = ROOT / EVIDENCE
    directory = ROOT / "artifacts/resource_probe/v255/cosine_cpu"
    if evidence.exists() or directory.exists():
        raise FileExistsError("CPU proof already exists; inspect instead of overwriting")
    before = fingerprints(source_paths())
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "execution_plan.json").open("x", encoding="utf-8") as stream:
        json.dump({"input_sha256": before, "updates": 72000, "constant_prefix": 36000,
                   "seed": 255001, "parameters": 4, "device": "cpu", "model_score_evaluated": False}, stream, indent=2)
    torch.set_num_threads(2)
    initial = torch.tensor([.1, -.2, .4, -.8], dtype=torch.float32)
    actual, reference, constant = [torch.nn.Parameter(initial.clone()) for _ in range(3)]
    optimizers = [torch.optim.Adam([p], lr=1e-5) for p in (actual, reference, constant)]
    recorder = RateRecorder(optimizers[0])
    generator = torch.Generator().manual_seed(255001)
    gradients = torch.randn(72000, 4, generator=generator)
    global_rng, local_rng = torch.random.get_rng_state().clone(), generator.get_state().clone()
    started = time.perf_counter()
    for update, gradient in enumerate(gradients, 1):
        for parameter, optimizer in zip((actual, reference, constant), optimizers, strict=True):
            optimizer.zero_grad(set_to_none=True)
            parameter.grad = gradient.clone()
            torch.nn.utils.clip_grad_norm_([parameter], max_norm=1.)
        # Independent direct reference, not the scheduling helper under test.
        optimizers[1].param_groups[0]["lr"] = (1e-5 if update <= 36000 else
            1e-5 * (.1 + .9 * .5 * (1. + math.cos(math.pi * (update - 36000) / 36000))))
        for optimizer in optimizers:
            optimizer.step()
        if not torch.equal(actual, reference):
            raise ValueError(f"manual-reference parameters diverged at {update}")
        for key in optimizers[0].state[actual]:
            if not torch.equal(optimizers[0].state[actual][key], optimizers[1].state[reference][key]):
                raise ValueError(f"manual-reference Adam state diverged: {key}, update{update}")
        if update <= 36000:
            if not torch.equal(actual, constant) or any(not torch.equal(
                    optimizers[0].state[actual][key], optimizers[2].state[constant][key])
                    for key in optimizers[0].state[actual]):
                raise ValueError(f"constant-control prefix diverged at {update}")
        if update % 12000 == 0:
            print(json.dumps({"cpu_verified_updates": update}), flush=True)
    rates = recorder.evidence()
    require_schedule(rates, 72000)
    rng_equal = torch.equal(global_rng, torch.random.get_rng_state()) and torch.equal(local_rng, generator.get_state())
    if not rng_equal or fingerprints(source_paths()) != before:
        raise RuntimeError("CPU proof RNG or source inputs changed")
    result = {"passed": True, "device": "cpu", "parameters": 4, "updates_checked": 72000,
              "constant_prefix_parameter_and_adam_state_exact_updates": 36000,
              "manual_schedule_parameter_and_adam_state_exact_updates": 72000,
              "global_and_local_rng_unchanged": True, "input_sha256": before, "actual_schedule": rates,
              "final_max_parameter_difference_vs_constant": float((actual - constant).abs().max().detach()),
              "elapsed_seconds": time.perf_counter() - started, "torch_version": torch.__version__,
              "model_score_evaluated": False,
              "caveat": "four-parameter CPU Adam proof only; actual model still requires initialization/full-prefix checks"}
    if result["final_max_parameter_difference_vs_constant"] <= 0:
        raise ValueError("schedule effect was not exercised by synthetic gradients")
    with evidence.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

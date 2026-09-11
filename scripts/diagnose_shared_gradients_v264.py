"""Frozen V257 training-only parameter-gradient decomposition; never optimizer steps."""
from __future__ import annotations

import json
import time

import torch
from audit_pure_neural_candidate import fingerprint
from diagnose_rms_proxy_v250 import rank_weights
from probe_pure_neural_shared_v257 import shared_name
from run_pure_neural_lr_v237 import ROOT, fingerprints
from run_storage_factorial_v254 import verify_bound_inputs, write_new
from train_pure_neural_rms_v239 import rms_score_loss
from train_pure_neural_snr_experts import PureNeuralLink, load_model_design, sample_snr

from oppods.data import ChannelMemmap, deterministic_split_indices

SOURCE = ROOT / "artifacts/pure_neural_v257/eight/shared8_steps72000"
PLAN = ROOT / "benchmarks/v264_parameter_gradient_plan.json"
OUTPUT = ROOT / "benchmarks/v264_parameter_gradient_diagnostic.json"


def contributions(logits, bits):
    signed = 2 * bits - 1
    normalized = logits / logits.square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
    scores = torch.sigmoid(signed * normalized / .5).mean(-1).flatten()
    weights = rank_weights(scores.detach())
    raw_bce = torch.nn.functional.softplus(-signed * logits).mean(-1).flatten()
    # Rank assignments are piecewise constant; retain the original global weights for every mask.
    return -(.7 / scores.numel() + .3 * weights) * scores + .05 * raw_bce / scores.numel()


def masks(snr):
    own = snr.flatten()
    partner = snr.flip(1).flatten()
    weak = (own < -15) & (partner < 0)
    strong = (own >= 0) & (partner < -15)
    return {"weak_both_negative": weak, "strong_with_weak_partner": strong, "rest": ~(weak | strong)}


def gradient_vector(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten()
                      for p, g in zip(parameters, gradients, strict=True)])


def cosine(a, b):
    denominator = a.norm() * b.norm()
    return float(torch.dot(a, b) / denominator) if denominator > 0 else None


def main():
    if PLAN.exists() or OUTPUT.exists():
        raise FileExistsError("registered diagnostic exists; do not overwrite or restart")
    files = fingerprint(SOURCE)
    audit_path = ROOT / "benchmarks/v257_eight_shared_prefix8_rms_72k_audit_offset2000.json"
    if files != json.loads(audit_path.read_text(encoding="utf-8"))["files"]:
        raise ValueError("V257 source mismatch")
    paths = [SOURCE / name for name in files]
    paths += [audit_path, ROOT / "ziliao/data_train/H_train.npz"]
    paths += [ROOT / name for name in (
        "scripts/diagnose_shared_gradients_v264.py", "scripts/diagnose_rms_proxy_v250.py",
        "scripts/probe_pure_neural_shared_v257.py", "scripts/train_pure_neural_snr_experts.py",
        "scripts/train_pure_neural_rms_v239.py", "src/oppods/data.py")]
    ids = deterministic_split_indices(100000, seed=1176)["train"][:400]
    plan = {"input_sha256": fingerprints(paths), "data_indices": ids.tolist(), "partition": "train",
            "split_seed": 1176, "seed": 26410, "batches": 4, "batch_size": 100,
            "snr_sampling": "original independent uniform [-20,20]", "device": "cuda", "gpu_cap": .25,
            "parameters": "actual unique shared input and first8 blocks in Tx/Rx parent-group0",
            "masks": "own<-15 and partner<0; own>=0 and partner<-15; complementary rest",
            "selection": "first400 training IDs, no screening by score or gradient", "optimizer_steps": 0,
            "interpretation": "descriptive local gradients; no guarantee of optimization benefit"}
    write_new(PLAN, plan)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.25)
    torch.manual_seed(26410)
    device = torch.device("cuda")
    link = PureNeuralLink(load_model_design(SOURCE / "modelDesign.py")).to(device).eval()
    link.load_submission(SOURCE)
    for parameter in link.parameters():
        parameter.requires_grad_(False)
    groups = {}
    for component in ("transmitter", "receiver"):
        expert = getattr(link, component).experts[0]
        groups[component] = [p for name, p in expert.named_parameters() if shared_name(component, name)]
        for parameter in groups[component]:
            parameter.requires_grad_(True)
    parameters = groups["transmitter"] + groups["receiver"]
    if len({id(p) for p in parameters}) != len(parameters):
        raise ValueError("duplicated tied parameters")
    split_at = sum(p.numel() for p in groups["transmitter"])
    generator = torch.Generator(device=device).manual_seed(26410)
    data = ChannelMemmap(ROOT / "ziliao/data_train/H_train.npz")
    records = []
    started = time.perf_counter()
    for start in range(0, 400, 100):
        channel = torch.from_numpy(data.read(ids[start:start+100])).to(device)
        bits = torch.randint(0, 2, (100, 2, 1152), generator=generator, device=device).float()
        snr = sample_snr(100, stage="calibrate", expert_index=None, device=device, generator=generator)
        logits = link(channel, bits, snr, generator=generator)
        parts = contributions(logits, bits)
        loss = rms_score_loss(logits, bits, loss_kind="rms_score", margin=.5, tail_weight=0., tail_fraction=.1,
                              score_temperature=.5, quantile_bandwidth=.025, score_bce_weight=.05,
                              score_fairness_weight=.3)
        torch.testing.assert_close(parts.sum(), loss, atol=2e-7, rtol=1e-6)
        regions = masks(snr)
        vectors = {name: gradient_vector(parts[mask].sum(), parameters) for name, mask in regions.items()}
        actual = gradient_vector(loss, parameters)
        torch.testing.assert_close(sum(vectors.values()), actual, atol=1e-6, rtol=1e-4)
        if not all(torch.isfinite(v).all() for v in [actual, *vectors.values()]):
            raise ValueError("nonfinite parameter gradient")
        record = {"batch": start // 100, "counts": {k: int(v.sum()) for k, v in regions.items()},
                  "loss": float(loss.detach()), "decomposition_max_error": float((sum(vectors.values())-actual).abs().max()),
                  "components": {}}
        for component, index in (("tx", slice(0, split_at)), ("rx", slice(split_at, None)), ("joint", slice(None))):
            v = {name: vector[index] for name, vector in vectors.items()}
            record["components"][component] = {
                "norms": {name: float(vector.norm()) for name, vector in v.items()},
                "weak_strong_cosine": cosine(v["weak_both_negative"], v["strong_with_weak_partner"]),
                "weak_rest_cosine": cosine(v["weak_both_negative"], v["rest"]),
                "weak_total_cosine": cosine(v["weak_both_negative"], actual[index]),
            }
        records.append(record)
        print(json.dumps(record), flush=True)
        del channel, bits, snr, logits, parts, loss, vectors, actual
    verify_bound_inputs(plan)
    if fingerprint(SOURCE) != files or any(p.grad is not None for p in link.parameters()):
        raise ValueError("source or accumulated gradients changed")
    write_new(OUTPUT, {"plan": str(PLAN), "model_files": files, "records": records,
                       "optimizer_steps": 0, "model_weights_saved": False,
                       "parameters_diagnosed": sum(p.numel() for p in parameters),
                       "elapsed_seconds": time.perf_counter()-started,
                       "caveat": "Four training batches only, not independent retraining or evidence of generalization."})


if __name__ == "__main__":
    main()

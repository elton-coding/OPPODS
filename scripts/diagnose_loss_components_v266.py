"""Training-only score/BCE parameter-gradient diagnostic; no optimizer or tuning."""
from __future__ import annotations

import json

import torch
from audit_pure_neural_candidate import fingerprint
from diagnose_shared_gradients_v264 import cosine, gradient_vector
from run_pure_neural_lr_v237 import ROOT, fingerprints
from run_storage_factorial_v254 import verify_bound_inputs, write_new
from train_pure_neural_rms_v239 import rms_score_loss
from train_pure_neural_snr_experts import PureNeuralLink, load_model_design, sample_snr

from oppods.data import ChannelMemmap, deterministic_split_indices

SOURCE = ROOT / "artifacts/pure_neural_v257/eight/shared8_steps72000"
PLAN = ROOT / "benchmarks/v266_loss_component_plan.json"
OUTPUT = ROOT / "benchmarks/v266_loss_component_diagnostic.json"


def losses(logits, bits):
    options = {"loss_kind": "rms_score", "margin": .5, "tail_weight": 0., "tail_fraction": .1,
               "score_temperature": .5, "quantile_bandwidth": .025, "score_fairness_weight": .3}
    score = rms_score_loss(logits, bits, score_bce_weight=0., **options)
    bce = .05 * torch.nn.functional.softplus(-(2 * bits - 1) * logits).mean()
    total = rms_score_loss(logits, bits, score_bce_weight=.05, **options)
    return score, bce, total


def main():
    if PLAN.exists() or OUTPUT.exists():
        raise FileExistsError("V266 diagnostic exists; no overwrite/restart")
    files = fingerprint(SOURCE)
    audit = ROOT / "benchmarks/v257_eight_shared_prefix8_rms_72k_audit_offset2000.json"
    if files != json.loads(audit.read_text(encoding="utf-8"))["files"]:
        raise ValueError("champion source changed")
    paths = [SOURCE / name for name in files] + [audit, ROOT / "ziliao/data_train/H_train.npz"]
    paths += [ROOT / name for name in (
        "scripts/diagnose_loss_components_v266.py", "scripts/diagnose_shared_gradients_v264.py",
        "scripts/train_pure_neural_rms_v239.py", "scripts/train_pure_neural_snr_experts.py",
        "scripts/diagnose_rms_proxy_v250.py", "scripts/probe_pure_neural_shared_v257.py",
        "src/oppods/data.py", "tests/test_loss_components_v266.py",
        "docs/experiments/loss-component-diagnostic-v266.md")]
    ids = deterministic_split_indices(100000, seed=1176)["train"][:400]
    plan = {"input_sha256": fingerprints(paths), "data_indices": ids.tolist(),
            "partition": "train", "seed": 26610, "batch_size": 100, "batches": 4,
            "optimizer_steps": 0, "gpu_cap": .4, "parameters": "all unique Tx and Rx parameters",
            "purpose": "describe existing score and weighted BCE gradient alignment; no weight search"}
    write_new(PLAN, plan)
    torch.set_num_threads(2)
    torch.manual_seed(26610)
    torch.cuda.set_per_process_memory_fraction(.4)
    device = torch.device("cuda")
    link = PureNeuralLink(load_model_design(SOURCE / "modelDesign.py")).to(device).eval()
    link.load_submission(SOURCE)
    for p in link.parameters():
        p.requires_grad_(False)
    tx = list(link.transmitter.parameters())
    rx = list(link.receiver.parameters())
    parameters = tx + rx
    assert len({id(p) for p in parameters}) == len(parameters)
    assert sum(p.numel() for p in parameters) == 72744448
    for p in parameters:
        p.requires_grad_(True)
    split_at = sum(p.numel() for p in tx)
    generator = torch.Generator(device=device).manual_seed(26610)
    data = ChannelMemmap(ROOT / "ziliao/data_train/H_train.npz")
    records = []
    for start in range(0, 400, 100):
        channel = torch.from_numpy(data.read(ids[start:start+100])).to(device)
        bits = torch.randint(0, 2, (100, 2, 1152), generator=generator, device=device).float()
        snr = sample_snr(100, stage="calibrate", expert_index=None, device=device, generator=generator)
        logits = link(channel, bits, snr, generator=generator)
        score, bce, total = losses(logits, bits)
        a, b, c = [gradient_vector(loss, parameters) for loss in (score, bce, total)]
        torch.testing.assert_close(a+b, c, atol=1e-6, rtol=1e-4)
        if not all(torch.isfinite(v).all() for v in (a, b, c)):
            raise ValueError("nonfinite gradients")
        row = {"batch": start//100, "score_loss": float(score.detach()),
               "weighted_bce": float(bce.detach()), "max_decomposition_error": float((a+b-c).abs().max()),
               "components": {}}
        for name, sl in (("tx", slice(0, split_at)), ("rx", slice(split_at, None)), ("joint", slice(None))):
            row["components"][name] = {"score_norm": float(a[sl].norm()), "weighted_bce_norm": float(b[sl].norm()),
                                       "score_bce_cosine": cosine(a[sl], b[sl]),
                                       "score_total_cosine": cosine(a[sl], c[sl])}
        records.append(row)
        print(json.dumps(row), flush=True)
        del channel, bits, snr, logits, score, bce, total, a, b, c
    verify_bound_inputs(plan)
    assert fingerprint(SOURCE) == files
    assert all(p.grad is None for p in link.parameters())
    write_new(OUTPUT, {"records": records, "model_files": files, "optimizer_steps": 0,
                       "weights_saved": False, "caveat": "Four training batches; Euclidean local gradients are not Adam updates or generalization evidence."})


if __name__ == "__main__":
    main()

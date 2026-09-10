"""CPU-only frozen-V250 diagnostic on training channels, never a candidate audit."""
from __future__ import annotations

import json
import time

import numpy as np
import torch
from audit_pure_neural_candidate import fingerprint
from run_pure_neural_lr_v237 import ROOT, fingerprints
from train_pure_neural_rms_v239 import MIN_RMS, rms_score_loss
from train_pure_neural_snr_experts import PureNeuralLink, load_model_design, sample_snr

from oppods.data import ChannelMemmap, deterministic_split_indices

SUBMISSION = "artifacts/pure_neural_v250/eight/steps72000"
CACHE = "artifacts/diagnostics/v250_rms_proxy/outputs_train400_seed27150.npz"
REPORT = "benchmarks/v250_rms_proxy_alignment_train400_seed27150.json"


def average_ranks(values):
    values = values.flatten()
    _, inverse, counts = torch.unique(values, sorted=True, return_inverse=True, return_counts=True)
    counts = counts.to(torch.float64)
    return (counts.cumsum(0) - .5 * (counts + 1.))[inverse]


def rank_correlation(a, b):
    a, b = average_ranks(a), average_ranks(b)
    a, b = a-a.mean(), b-b.mean()
    denominator = a.norm()*b.norm()
    return float((a*b).sum()/denominator) if denominator > 0 else None


def rank_weights(scores, fraction=.1, bandwidth=.025):
    if scores.ndim != 1 or scores.numel() == 0 or bandwidth <= 0 or not 0 <= fraction <= 1:
        raise ValueError("rank weights need nonempty flat scores and valid quantile parameters")
    ranks = torch.arange(scores.numel(), dtype=scores.dtype, device=scores.device)
    target = fraction * max(0, scores.numel()-1)
    scale = max(1., bandwidth * scores.numel())
    weights = torch.softmax(-.5 * ((ranks-target)/scale).square(), 0)
    result = torch.empty_like(scores)
    result[torch.sort(scores).indices] = weights
    return result


def analyze(logits, bits, snr, temperature):
    if (logits.shape != bits.shape or logits.ndim != 3 or logits.shape[-1] != 1152
            or snr.shape != logits.shape[:2] or temperature <= 0
            or not torch.isfinite(logits).all() or not torch.isfinite(snr).all()
            or not ((bits == 0) | (bits == 1)).all()):
        raise ValueError("diagnostic requires finite full-payload logits, binary bits and aligned SNR")
    x = logits.detach().clone().requires_grad_(True)
    normalized = x / x.square().mean(-1, keepdim=True).clamp_min(MIN_RMS**2).sqrt()
    signed = (2.*bits-1.)
    soft = torch.sigmoid(signed*normalized/temperature).mean(-1).flatten()
    hard = ((x.detach() >= 0) == (bits >= .5)).float().mean(-1).flatten()
    weights = rank_weights(soft)
    proxy_p10 = (weights*soft).sum()
    hard_p10 = torch.quantile(hard, .1)
    hard_tail = hard <= hard_p10
    soft_tail = soft.detach() <= torch.quantile(soft.detach(), .1)
    near_hard_p10 = (hard-hard_p10).abs() <= .01
    low_snr = snr.flatten() < -10.
    loss = rms_score_loss(x, bits, loss_kind="rms_score", margin=.5, tail_weight=0.,
                          tail_fraction=.1, score_temperature=temperature, quantile_bandwidth=.025,
                          score_bce_weight=.05, score_fairness_weight=.3, valid_lengths=None)
    reconstructed = -(.7*soft.mean()+.3*proxy_p10)+.05*torch.nn.functional.softplus(-signed*x).mean()
    torch.testing.assert_close(loss.detach(), reconstructed.detach(), atol=2e-7, rtol=0)
    total_gradient = torch.autograd.grad(loss, x)[0]
    fairness_gradient = torch.autograd.grad(-proxy_p10, x)[0]
    assert torch.isfinite(total_gradient).all() and torch.isfinite(fairness_gradient).all()

    def gradient_mass(gradient):
        mass = gradient.abs().sum(-1).flatten()
        total = mass.sum()
        return {"absolute_gradient_sum": float(total),
                "fraction_at_hard_p10_plus_minus_one_point": float(mass[near_hard_p10].sum()/total) if total > 0 else None,
                "fraction_at_own_snr_below_minus10": float(mass[low_snr].sum()/total) if total > 0 else None}

    return {"ue_observations": hard.numel(), "temperature": temperature,
            "hard_efficiency_training_diagnostic_only": float(hard.mean()*100),
            "hard_p10_training_diagnostic_only": float(hard_p10*100),
            "soft_mean_proxy": float(soft.detach().mean()*100), "soft_p10_proxy": float(proxy_p10.detach()*100),
            "average_rank_correlation": rank_correlation(hard, soft.detach()),
            "hard_tail_count_including_ties": int(hard_tail.sum()), "soft_tail_count_including_ties": int(soft_tail.sum()),
            "tail_intersection_count": int((hard_tail & soft_tail).sum()),
            "hard_tail_recalled_by_soft_tail": float((hard_tail & soft_tail).sum()/hard_tail.sum()),
            "hard_p10_plus_minus_one_point_count": int(near_hard_p10.sum()),
            "quantile_rank_weight_mass_at_hard_p10_plus_minus_one_point": float(weights[near_hard_p10].sum()),
            "quantile_rank_weight_mass_at_own_snr_below_minus10": float(weights[low_snr].sum()),
            "actual_rms_training_objective": float(loss.detach()),
            "reconstructed_objective_difference": float((reconstructed-loss).detach()),
            "total_loss_gradient_mass": gradient_mass(total_gradient),
            "p10_only_gradient_mass": gradient_mass(fairness_gradient)}


def main():
    output, cache = ROOT / REPORT, ROOT / CACHE
    if output.exists() or cache.exists():
        raise FileExistsError("diagnostic evidence exists; no overwrite or implicit rerun")
    torch.set_num_threads(2)
    paths = [ROOT / SUBMISSION / name for name in ("modelDesign.py", "encoder.pth", "transmitter.pth", "receiver.pth")]
    names = ["scripts/diagnose_rms_proxy_v250.py", "scripts/train_pure_neural_rms_v239.py",
             "scripts/train_pure_neural_snr_experts.py", "scripts/audit_pure_neural_candidate.py",
             "scripts/run_pure_neural_lr_v237.py", "src/oppods/data.py", "ziliao/data_train/H_train.npz",
             "benchmarks/v250_eight_rms_72k_audit_offset2000.json"]
    paths += [ROOT / name for name in names]
    before = fingerprints(paths)
    audit = json.loads((ROOT / names[-1]).read_text(encoding="utf-8"))
    if audit["label"] != "v250_eight_rms_72k" or fingerprint(ROOT / SUBMISSION) != audit["files"]:
        raise ValueError("frozen V250 weights differ from audited champion")
    started = time.perf_counter()
    data = ChannelMemmap(ROOT / "ziliao/data_train/H_train.npz")
    ids = deterministic_split_indices(len(data), seed=1176)["train"][:400]
    assert len(ids) == len(np.unique(ids)) == 400
    module = load_model_design(ROOT / SUBMISSION / "modelDesign.py")
    torch.manual_seed(27150)
    link = PureNeuralLink(module).eval()
    link.load_submission(ROOT / SUBMISSION)
    generator = torch.Generator(device="cpu").manual_seed(27150)
    logits_list, bits_list, snr_list = [], [], []
    for start in range(0, 400, 100):
        channel = torch.from_numpy(data.read(ids[start:start+100]))
        bits = torch.randint(0, 2, (100,2,1152), generator=generator).float()
        snr = sample_snr(100, stage="calibrate", expert_index=None, device=torch.device("cpu"), generator=generator)
        with torch.no_grad():
            logits = link(channel, bits, snr, generator=generator)
        assert logits.shape == bits.shape and torch.isfinite(logits).all()
        logits_list.append(logits)
        bits_list.append(bits)
        snr_list.append(snr)
        print(json.dumps({"cpu_training_channels_diagnosed": start+100, "of": 400}), flush=True)
    logits, bits, snr = torch.cat(logits_list), torch.cat(bits_list), torch.cat(snr_list)
    cases = []
    for temperature in (.5, .25):
        for start in range(0, 400, 100):
            cases.append({"grouping": "original_batch100", "channel_start": start,
                          **analyze(logits[start:start+100],bits[start:start+100],snr[start:start+100],temperature)})
        cases.append({"grouping": "pooled400_not_v259_training", "channel_start": 0,
                      **analyze(logits,bits,snr,temperature)})
    if before != fingerprints(paths) or fingerprint(ROOT / SUBMISSION) != audit["files"]:
        raise RuntimeError("frozen diagnostic inputs changed")
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("xb") as stream:
        np.savez_compressed(stream, logits=logits.numpy(), bits=bits.numpy().astype(np.uint8),
                            snr=snr.numpy(), data_index=ids, split_seed=np.int32(1176), noise_seed=np.int32(27150),
                            data_partition=np.array("train"))
    record = {"purpose": "frozen-model training-partition loss diagnostic, not candidate/confirmation/online score",
              "device": "cpu", "model_label": audit["label"], "model_files": audit["files"], "input_sha256": before,
              "cache_sha256": fingerprints([cache]), "channels": 400, "unique_training_channel_ids": 400,
              "split_seed": 1176, "random_seed": 27150, "temperatures_fixed_in_advance": [.5,.25],
              "all_model_gradients_absent": all(p.grad is None for p in link.parameters()),
              "model_weights_saved": False, "elapsed_seconds": time.perf_counter()-started, "cases": cases,
              "caveat": "Training data and reused frozen model; descriptive only. Gradients are with respect to logits, not model parameters. Neither rank correlation nor gradient concentration proves a training gain. Pooled observations are not independent-channel repetitions or V259 evaluation."}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(record,stream,indent=2,allow_nan=False)
    print(json.dumps({"report": str(output), "cases": len(cases), "elapsed_seconds": record["elapsed_seconds"]}),flush=True)


if __name__ == "__main__":
    main()

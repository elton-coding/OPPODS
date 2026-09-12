"""Matched-randomness train/validation diagnostic of two frozen completed models."""
from __future__ import annotations

import gc
import json

import numpy as np
import torch
from audit_pure_neural_candidate import fingerprint
from compare_paired_holdout import metrics
from run_pure_neural_lr_v237 import ROOT, fingerprints
from run_storage_factorial_v254 import verify_bound_inputs, write_new
from train_pure_neural_rms_v239 import rms_score_loss
from train_pure_neural_snr_experts import PureNeuralLink, load_model_design, sample_snr

from oppods.data import ChannelMemmap, deterministic_split_indices

SOURCES = {"V269": ("artifacts/pure_neural_v269/eight/lr3e5_steps72000", "v269_shared8_rms_lr3e5_72k"),
           "V273": ("artifacts/pure_neural_v273/eight/steps144000", "v273_shared8_rms_144k")}
PLAN = ROOT / "benchmarks/v274_train_validation_plan.json"
OUTPUT = ROOT / "benchmarks/v274_train_validation_diagnostic.json"
SEEDS = [27421, 27422, 27423]


def proxy_summary(hard, soft):
    hard, soft = np.asarray(hard, dtype=float), np.asarray(soft, dtype=float)
    if hard.shape != soft.shape or hard.ndim != 1 or hard.size < 2:
        raise ValueError("paired one-dimensional scores required")
    if not np.isfinite(hard).all() or not np.isfinite(soft).all():
        raise ValueError("nonfinite scores")
    tail = hard <= np.quantile(hard, .1)
    proxy_tail = soft <= np.quantile(soft, .1)
    near = np.abs(hard - np.quantile(hard, .1)) <= 1.
    corr = float(np.corrcoef(hard, soft)[0, 1]) if hard.std() > 0 and soft.std() > 0 else None
    return {"proxy_metrics": metrics(soft), "pearson": corr,
            "hard_tail_rows": int(tail.sum()), "proxy_tail_rows": int(proxy_tail.sum()),
            "tail_recall": float(proxy_tail[tail].mean()),
            "tail_precision": float(tail[proxy_tail].mean()),
            "mean_proxy_minus_hard": float((soft-hard).mean()),
            "near_hard_p10_rows": int(near.sum()),
            "near_p10_proxy_minus_hard": float((soft-hard)[near].mean()) if near.any() else None}


def main():
    if PLAN.exists() or OUTPUT.exists():
        raise FileExistsError("diagnostic exists; no overwrite")
    split = deterministic_split_indices(100000, seed=1176)
    ids = {name: split[name][:2000] for name in ("train", "validation")}
    assert not np.intersect1d(*ids.values()).size
    paths, files = [], {}
    for name, (directory, label) in SOURCES.items():
        path = ROOT / f"benchmarks/{label}_audit_offset2000.json"
        files[name] = fingerprint(ROOT / directory)
        if files[name] != json.loads(path.read_text(encoding="utf-8"))["files"]:
            raise ValueError("frozen model mismatch")
        paths += [path, *[ROOT / directory / filename for filename in files[name]]]
    paths += [ROOT / name for name in ("scripts/diagnose_train_validation_v274.py",
        "scripts/train_pure_neural_snr_experts.py", "scripts/train_pure_neural_rms_v239.py",
        "tests/test_score_proxy_v274.py", "docs/experiments/score-proxy-v274.md", "scripts/compare_paired_holdout.py",
        "src/oppods/data.py", "ziliao/data_train/H_train.npz")]
    plan = {"input_sha256": fingerprints(paths), "ids": {k: v.tolist() for k, v in ids.items()},
            "seeds": SEEDS, "batch_size": 100, "samples_per_partition": 2000,
            "model_updates": False, "partition_selection": "first2000 fixed split1176; no score screening",
            "same_random_stream": "same bits/SNR/noise at matched positions across models and partitions",
            "caveat": "different channel IDs across partitions; validation already used for checkpoint selection"}
    write_new(PLAN, plan)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.25)
    device = torch.device("cuda")
    data = ChannelMemmap(ROOT / "ziliao/data_train/H_train.npz")
    records = {}
    for name, (directory, _) in SOURCES.items():
        torch.manual_seed(27420)
        link = PureNeuralLink(load_model_design(ROOT / directory / "modelDesign.py")).to(device).eval()
        link.load_submission(ROOT / directory)
        records[name] = {}
        for partition, indices in ids.items():
            rows = []
            for seed in SEEDS:
                generator = torch.Generator(device=device).manual_seed(seed)
                scores, bces, snrs, proxies, batches = [], [], [], [], []
                for start in range(0, 2000, 100):
                    channel = torch.from_numpy(data.read(indices[start:start+100])).to(device)
                    bits = torch.randint(0, 2, (100, 2, 1152), generator=generator, device=device).float()
                    snr = sample_snr(100, stage="calibrate", expert_index=None, device=device, generator=generator)
                    with torch.no_grad():
                        logits = link(channel, bits, snr, generator=generator)
                        assert logits.shape == bits.shape and torch.isfinite(logits).all()
                        score = ((logits >= 0) == (bits >= .5)).float().mean(-1) * 100
                        bce = torch.nn.functional.softplus(-(2 * bits - 1) * logits).mean(-1)
                        rms = logits.square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
                        proxy = torch.sigmoid((2 * bits - 1) * logits / rms / .5).mean(-1) * 100
                        objective = rms_score_loss(logits, bits, loss_kind="rms_score",
                            margin=.5, tail_weight=0., tail_fraction=.1,
                            score_temperature=.5, quantile_bandwidth=.025,
                            score_bce_weight=.05, score_fairness_weight=.3)
                        batches.append({"start": start, "training_objective": float(objective),
                            "hard_final": metrics(score.cpu().numpy().flatten().astype(float))["final"]})
                    proxies.append(proxy.cpu().numpy().flatten())
                    scores.append(score.cpu().numpy().flatten())
                    bces.append(bce.cpu().numpy().flatten())
                    snrs.append(snr.cpu().numpy().flatten())
                score, bce, snr = np.concatenate(scores), np.concatenate(bces), np.concatenate(snrs)
                proxy = np.concatenate(proxies)
                row = {"seed": seed, **metrics(score.astype(float)), "bce": float(bce.mean()),
                       "proxy_diagnostic": proxy_summary(score, proxy), "batches": batches,
                       "own_snr_slices": {str(lo): {"rows": int(((snr >= lo) & (snr < lo+5)).sum()),
                            "mean": float(score[(snr >= lo) & (snr < lo+5)].mean())}
                            for lo in range(-20, 20, 5)}}
                rows.append(row)
                print(json.dumps({"model": name, "partition": partition, **row}), flush=True)
            records[name][partition] = rows
        del link, channel, logits
        gc.collect()
        torch.cuda.empty_cache()
    verify_bound_inputs(plan)
    write_new(OUTPUT, {"records": records, "model_files": files, "plan": str(PLAN),
        "diagnostic_only": True, "optimizer_steps": 0,
        "caveat": "Not official batch1 audit; fixed2000 channel samples, same noise replicas not new channels; no overfitting proof from a small gap alone."})


if __name__ == "__main__":
    main()

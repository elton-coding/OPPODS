from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from oppods.adaptive_mcs_link import AdaptiveMCSMUMIMOLink, select_mcs
from oppods.data import ChannelMemmap, deterministic_split_indices
from oppods.denoised_link import DenoisedSparseMUMIMOLink
from oppods.metrics import per_sample_score, summarize_scores

PROFILES: dict[str, tuple[float, float, float, float]] = {
    "p0": (-10.0, -5.0, 1.0, 8.0),
    "p1": (-9.0, -4.0, 2.0, 9.0),
    "p2": (-8.0, -3.0, 3.0, 10.0),
    "p3": (-7.0, -2.0, 4.0, 11.0),
    "p4": (-6.0, -1.0, 5.0, 12.0),
    "p5": (-5.0, 0.0, 6.0, 13.0),
    "p6": (-8.0, -1.0, 6.0, 14.0),
    "p7": (-6.0, 1.0, 8.0, 15.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired evaluation of SNR-adaptive modulation")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/sparse_denoiser_task.pt"))
    parser.add_argument("--data", type=Path, default=Path("ziliao/data_train/H_train.npz"))
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1176)
    parser.add_argument("--tail-threshold", type=float, default=-10.5)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    data = ChannelMemmap(args.data)
    indices = deterministic_split_indices(len(data), seed=args.seed)["validation"][: args.samples]
    count = len(indices)
    rng = np.random.default_rng(args.seed)
    bits_np = rng.integers(0, 2, (count, 2, 1152), dtype=np.int8)
    snr_np = rng.uniform(-20.0, 20.0, (count, 2)).astype(np.float32)
    variants: dict[str, DenoisedSparseMUMIMOLink] = {"baseline": DenoisedSparseMUMIMOLink().to(device)}
    variants.update({name: AdaptiveMCSMUMIMOLink(profile).to(device) for name, profile in PROFILES.items()})
    for link in variants.values():
        link.load_denoiser_checkpoint(args.checkpoint)
        link.eval()

    chunks: dict[str, list[np.ndarray]] = {name: [] for name in variants}
    for start in range(0, count, args.batch_size):
        end = min(start + args.batch_size, count)
        channel = torch.from_numpy(data.read(indices[start:end])).to(device)
        bits = torch.from_numpy(bits_np[start:end]).to(device=device, dtype=torch.float32)
        snr = torch.from_numpy(snr_np[start:end]).to(device)
        for name, link in variants.items():
            generator = torch.Generator(device=device).manual_seed(args.seed + start)
            llr = link(channel, bits, snr, generator=generator)
            if name == "baseline":
                lengths = torch.where(snr < args.tail_threshold, 1, 1152)
            else:
                lengths = 144 * select_mcs(snr, PROFILES[name])
                lengths = torch.where(snr < args.tail_threshold, 1, lengths)
            batch_scores = torch.empty_like(snr)
            for user in range(2):
                for length in torch.unique(lengths[:, user]).tolist():
                    mask = lengths[:, user] == length
                    batch_scores[mask, user] = per_sample_score(bits[mask, user], llr[mask, user, :length])
            chunks[name].append(batch_scores.cpu().numpy())

    all_scores = {name: np.concatenate(values) for name, values in chunks.items()}
    baseline = all_scores["baseline"]
    result: dict[str, object] = {"samples": count, "seed": args.seed, "variants": {}}
    for name, values in all_scores.items():
        summary = summarize_scores(values)
        result["variants"][name] = {
            "thresholds": PROFILES.get(name),
            "efficiency": summary.efficiency,
            "fairness": summary.fairness,
            "final": summary.final,
            "mean_sample_delta": float(np.mean(values - baseline)),
            "p10_sample_delta": float(np.percentile(values - baseline, 10)),
            "win_fraction": float(np.mean(values > baseline)),
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

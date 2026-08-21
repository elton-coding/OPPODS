from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from oppods.data import ChannelMemmap, deterministic_split_indices
from oppods.denoised_link import DenoisedSparseMUMIMOLink
from oppods.metrics import per_sample_score, summarize_scores
from oppods.paired_pilot_link import PairedPilotMUMIMOLink

PAIRED_PREFIXES = (1, 132, 264, 396, 528, 660, 792, 924, 1056)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search layered-QAM prefix rules for the Walsh-pilot profile")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/sparse_denoiser_task.pt"))
    parser.add_argument("--data", type=Path, default=Path("ziliao/data_train/H_train.npz"))
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1176)
    parser.add_argument("--profile-threshold", type=float, default=14.0)
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
    blind = DenoisedSparseMUMIMOLink().to(device)
    paired = PairedPilotMUMIMOLink(
        1.5,
        separator="tri3",
        covariance_loading_scale=2.0,
        gain_refinement_iterations=4,
        gain_refinement_min_snr=-5.0,
        gain_refinement_rate=0.75,
    ).to(device)
    blind.load_denoiser_checkpoint(args.checkpoint)
    paired.load_denoiser_checkpoint(args.checkpoint)
    blind.eval()
    paired.eval()
    paired_chunks: dict[int, list[np.ndarray]] = {prefix: [] for prefix in PAIRED_PREFIXES}
    blind_one_chunks: list[np.ndarray] = []
    blind_full_chunks: list[np.ndarray] = []
    for start in range(0, count, args.batch_size):
        end = min(start + args.batch_size, count)
        channel = torch.from_numpy(data.read(indices[start:end])).to(device)
        bits = torch.from_numpy(bits_np[start:end]).to(device=device, dtype=torch.float32)
        snr = torch.from_numpy(snr_np[start:end]).to(device)
        blind_generator = torch.Generator(device=device).manual_seed(args.seed + start)
        paired_generator = torch.Generator(device=device).manual_seed(args.seed + start)
        blind_llr = blind(channel, bits, snr, generator=blind_generator)
        paired_llr, _ = paired(channel, bits, snr, generator=paired_generator)
        for prefix in PAIRED_PREFIXES:
            paired_chunks[prefix].append(
                torch.stack(
                    [per_sample_score(bits[:, user], paired_llr[:, user, :prefix]) for user in range(2)], dim=1
                )
                .cpu()
                .numpy()
            )
        blind_one_chunks.append(
            torch.stack([per_sample_score(bits[:, user], blind_llr[:, user, :1]) for user in range(2)], dim=1)
            .cpu()
            .numpy()
        )
        blind_full_chunks.append(
            torch.stack([per_sample_score(bits[:, user], blind_llr[:, user]) for user in range(2)], dim=1)
            .cpu()
            .numpy()
        )
    paired_scores = {prefix: np.concatenate(parts) for prefix, parts in paired_chunks.items()}
    blind_one = np.concatenate(blind_one_chunks)
    blind_full = np.concatenate(blind_full_chunks)
    use_paired = snr_np.max(axis=1) < args.profile_threshold
    candidates = []
    for low_threshold in np.arange(-16.0, -13.99, 0.5):
        blind_dynamic = np.where(snr_np < low_threshold, blind_one, blind_full)
        for middle_prefix in PAIRED_PREFIXES[1:-1]:
            for high_threshold in np.arange(low_threshold + 0.5, 10.01, 0.5):
                paired_dynamic = np.where(
                    snr_np < low_threshold,
                    paired_scores[1],
                    np.where(snr_np < high_threshold, paired_scores[middle_prefix], paired_scores[1056]),
                )
                selected = np.where(use_paired[:, None], paired_dynamic, blind_dynamic)
                summary = summarize_scores(selected)
                candidates.append(
                    {
                        "low_threshold_db": float(low_threshold),
                        "middle_prefix": middle_prefix,
                        "high_threshold_db": float(high_threshold),
                        "efficiency": summary.efficiency,
                        "fairness": summary.fairness,
                        "final": summary.final,
                    }
                )
    print(
        json.dumps(
            {
                "samples": count,
                "seed": args.seed,
                "profile_threshold_db": args.profile_threshold,
                "best": max(candidates, key=lambda item: item["final"]),
                "top": sorted(candidates, key=lambda item: item["final"], reverse=True)[:12],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from oppods.data import ChannelMemmap, deterministic_split_indices
from oppods.denoised_link import DenoisedSparseMUMIMOLink
from oppods.metrics import per_sample_score, summarize_scores
from oppods.reserved_pilot_link import ReservedPilotMUMIMOLink


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare blind and reserved-pilot transmission profiles")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/sparse_denoiser_task.pt"))
    parser.add_argument("--data", type=Path, default=Path("ziliao/data_train/H_train.npz"))
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1176)
    parser.add_argument("--pilot-amplitude", type=float, default=2.0)
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
    blind = DenoisedSparseMUMIMOLink().to(device)
    reserved = ReservedPilotMUMIMOLink(args.pilot_amplitude).to(device)
    blind.load_denoiser_checkpoint(args.checkpoint)
    reserved.load_denoiser_checkpoint(args.checkpoint)
    blind.eval()
    reserved.eval()
    blind_generator = torch.Generator(device=device).manual_seed(args.seed)
    reserved_generator = torch.Generator(device=device).manual_seed(args.seed)
    score_batches: dict[str, list[np.ndarray]] = {
        "blind": [],
        "blind_one": [],
        "reserved": [],
        "reserved_one": [],
    }
    for start in range(0, count, args.batch_size):
        end = min(start + args.batch_size, count)
        channel = torch.from_numpy(data.read(indices[start:end])).to(device)
        bits = torch.from_numpy(bits_np[start:end]).to(device=device, dtype=torch.float32)
        snr = torch.from_numpy(snr_np[start:end]).to(device)
        blind_llr = blind(channel, bits, snr, generator=blind_generator)
        reserved_llr, _ = reserved(channel, bits, snr, generator=reserved_generator)
        for name, llr in (("blind", blind_llr), ("reserved", reserved_llr)):
            score_batches[name].append(
                torch.stack([per_sample_score(bits[:, user], llr[:, user]) for user in range(2)], dim=1).cpu().numpy()
            )
            score_batches[f"{name}_one"].append(
                torch.stack([per_sample_score(bits[:, user], llr[:, user, :1]) for user in range(2)], dim=1)
                .cpu()
                .numpy()
            )
    scores = {name: np.concatenate(batches) for name, batches in score_batches.items()}
    low_snr = snr_np < args.tail_threshold

    candidates = []
    for high_threshold in np.arange(-10.0, 15.01, 0.5):
        for low_threshold in (-20.0, -15.0, -10.0, -5.0, 0.0):
            maximum = snr_np.max(axis=1)
            minimum = snr_np.min(axis=1)
            use_reserved = (maximum < high_threshold) & (maximum >= low_threshold)
            use_reserved_both_in_range = (maximum < high_threshold) & (minimum >= low_threshold)
            for rule, profile in (
                ("max_range", use_reserved),
                ("both_range", use_reserved_both_in_range),
            ):
                selected = np.where(profile[:, None], scores["reserved"], scores["blind"])
                selected_one = np.where(profile[:, None], scores["reserved_one"], scores["blind_one"])
                dynamic = np.where(low_snr, selected_one, selected)
                summary = summarize_scores(dynamic)
                candidates.append(
                    {
                        "rule": rule,
                        "low_threshold_db": low_threshold,
                        "high_threshold_db": float(high_threshold),
                        "reserved_fraction": float(profile.mean()),
                        "efficiency": summary.efficiency,
                        "fairness": summary.fairness,
                        "final": summary.final,
                    }
                )
    best = max(candidates, key=lambda item: item["final"])
    blind_dynamic = np.where(low_snr, scores["blind_one"], scores["blind"])
    blind_summary = summarize_scores(blind_dynamic)
    print(
        json.dumps(
            {
                "samples": count,
                "blind_dynamic": {
                    "efficiency": blind_summary.efficiency,
                    "fairness": blind_summary.fairness,
                    "final": blind_summary.final,
                },
                "best_hybrid": best,
                "top_hybrid_rules": sorted(candidates, key=lambda item: item["final"], reverse=True)[:10],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

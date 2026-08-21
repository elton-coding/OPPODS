from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from oppods.data import ChannelMemmap, deterministic_split_indices
from oppods.metrics import per_sample_score, summarize_scores
from oppods.pilot_link import PilotAidedMUMIMOLink


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate superimposed-pilot receiver observability")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/sparse_denoiser_task.pt"))
    parser.add_argument("--data", type=Path, default=Path("ziliao/data_train/H_train.npz"))
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1176)
    parser.add_argument("--pilot-power", type=float, default=0.1)
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
    link = PilotAidedMUMIMOLink(args.pilot_power).to(device)
    link.load_denoiser_checkpoint(args.checkpoint)
    link.eval()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    score_batches = []
    selection = []
    for start in range(0, count, args.batch_size):
        end = min(start + args.batch_size, count)
        channel = torch.from_numpy(data.read(indices[start:end])).to(device)
        bits = torch.from_numpy(bits_np[start:end]).to(device=device, dtype=torch.float32)
        snr = torch.from_numpy(snr_np[start:end]).to(device)
        llr, correct_selection = link(channel, bits, snr, generator=generator)
        score_batches.append(
            torch.stack([per_sample_score(bits[:, user], llr[:, user]) for user in range(2)], dim=1).cpu().numpy()
        )
        selection.append(correct_selection.cpu().numpy())
    scores = np.concatenate(score_batches)
    summary = summarize_scores(scores)
    print(
        json.dumps(
            {
                "samples": count,
                "pilot_power": args.pilot_power,
                "efficiency": summary.efficiency,
                "fairness": summary.fairness,
                "final": summary.final,
                "pilot_identity_accuracy": float(np.concatenate(selection, axis=0).mean()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

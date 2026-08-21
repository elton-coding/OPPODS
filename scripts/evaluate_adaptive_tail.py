from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from oppods.adaptive_tail_link import AdaptiveTailMUMIMOLink
from oppods.data import ChannelMemmap, deterministic_split_indices
from oppods.denoised_link import DenoisedSparseMUMIMOLink
from oppods.metrics import per_sample_score, summarize_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired evaluation of an extreme-SNR repetition profile")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/sparse_denoiser_task.pt"))
    parser.add_argument("--data", type=Path, default=Path("ziliao/data_train/H_train.npz"))
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1176)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[-18.0, -16.0, -14.0, -12.0, -10.0])
    parser.add_argument("--code-lengths", type=int, nargs="+", default=[4, 12, 24, 48, 72, 144])
    parser.add_argument("--score-tail-threshold", type=float, default=-10.5)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def scores(
    bits: torch.Tensor,
    llr: torch.Tensor,
    snr: torch.Tensor,
    threshold: float,
    profile_threshold: float | None = None,
    code_length: int = 1,
) -> np.ndarray:
    full = torch.stack([per_sample_score(bits[:, user], llr[:, user]) for user in range(2)], dim=1)
    one = torch.stack([per_sample_score(bits[:, user], llr[:, user, :1]) for user in range(2)], dim=1)
    dynamic = torch.where(snr < threshold, one, full)
    if profile_threshold is not None:
        coded = torch.stack(
            [per_sample_score(bits[:, user], llr[:, user, :code_length]) for user in range(2)], dim=1
        )
        dynamic = torch.where(snr < profile_threshold, coded, dynamic)
    return dynamic.cpu().numpy()


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
    profiles: dict[str, tuple[float, int] | None] = {"baseline": None}
    for threshold in args.thresholds:
        for code_length in args.code_lengths:
            name = f"tail_{threshold:g}_k{code_length}"
            variants[name] = AdaptiveTailMUMIMOLink(
                tail_threshold_db=threshold,
                code_length=code_length,
            ).to(device)
            profiles[name] = (threshold, code_length)
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
            profile = profiles[name]
            profile_threshold, code_length = profile if profile is not None else (None, 1)
            chunks[name].append(
                scores(bits, llr, snr, args.score_tail_threshold, profile_threshold, code_length)
            )

    all_scores = {name: np.concatenate(values) for name, values in chunks.items()}
    baseline = all_scores["baseline"]
    result: dict[str, object] = {"samples": count, "seed": args.seed, "variants": {}}
    for name, values in all_scores.items():
        summary = summarize_scores(values)
        result["variants"][name] = {
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

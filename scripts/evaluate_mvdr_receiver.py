from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from oppods.data import ChannelMemmap, deterministic_split_indices
from oppods.denoised_link import DenoisedSparseMUMIMOLink
from oppods.metrics import per_sample_score, summarize_scores
from oppods.mvdr_receiver import MVDRAnalyticReceiver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired evaluation of blind matched and MVDR receivers")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/sparse_denoiser_task.pt"))
    parser.add_argument("--data", type=Path, default=Path("ziliao/data_train/H_train.npz"))
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1176)
    parser.add_argument("--trace-loadings", type=float, nargs="+", default=[0.0, 0.03, 0.1, 0.3, 1.0])
    parser.add_argument("--noise-loading", type=float, default=1.0)
    parser.add_argument("--tail-threshold", type=float, default=-10.5)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def dynamic_scores(bits: torch.Tensor, llr: torch.Tensor, snr: torch.Tensor, threshold: float) -> np.ndarray:
    full = torch.stack([per_sample_score(bits[:, user], llr[:, user]) for user in range(2)], dim=1)
    one = torch.stack([per_sample_score(bits[:, user], llr[:, user, :1]) for user in range(2)], dim=1)
    return torch.where(snr < threshold, one, full).cpu().numpy()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    data = ChannelMemmap(args.data)
    indices = deterministic_split_indices(len(data), seed=args.seed)["validation"][: args.samples]
    count = len(indices)
    rng = np.random.default_rng(args.seed)
    bits_np = rng.integers(0, 2, (count, 2, 1152), dtype=np.int8)
    snr_np = rng.uniform(-20.0, 20.0, (count, 2)).astype(np.float32)

    matched = DenoisedSparseMUMIMOLink().to(device)
    matched.load_denoiser_checkpoint(args.checkpoint)
    matched.eval()
    variants: dict[str, DenoisedSparseMUMIMOLink] = {"matched": matched}
    for trace_loading in args.trace_loadings:
        link = DenoisedSparseMUMIMOLink().to(device)
        link.load_denoiser_checkpoint(args.checkpoint)
        link.receiver = MVDRAnalyticReceiver(
            group_size=12,
            bits_per_symbol=8,
            interference_scale=1.0,
            use_agc=True,
            decision_directed_iterations=15,
            central_boost=-0.5,
            trace_loading=trace_loading,
            noise_loading=args.noise_loading,
        ).to(device)
        link.eval()
        variants[f"mvdr_trace_{trace_loading:g}"] = link

    score_batches: dict[str, list[np.ndarray]] = {name: [] for name in variants}
    for start in range(0, count, args.batch_size):
        end = min(start + args.batch_size, count)
        channel = torch.from_numpy(data.read(indices[start:end])).to(device)
        bits = torch.from_numpy(bits_np[start:end]).to(device=device, dtype=torch.float32)
        snr = torch.from_numpy(snr_np[start:end]).to(device)
        for name, link in variants.items():
            generator = torch.Generator(device=device).manual_seed(args.seed + start)
            llr = link(channel, bits, snr, generator=generator)
            score_batches[name].append(dynamic_scores(bits, llr, snr, args.tail_threshold))

    results: dict[str, object] = {"samples": count, "seed": args.seed, "variants": {}}
    scores = {name: np.concatenate(chunks) for name, chunks in score_batches.items()}
    baseline = scores["matched"]
    for name, values in scores.items():
        summary = summarize_scores(values)
        results["variants"][name] = {
            "efficiency": summary.efficiency,
            "fairness": summary.fairness,
            "final": summary.final,
            "mean_sample_delta": float((values - baseline).mean()),
            "win_fraction": float((values > baseline).mean()),
        }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

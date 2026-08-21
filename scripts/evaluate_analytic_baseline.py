from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from oppods.analytic_baseline import AnalyticMUMIMOLink
from oppods.constants import DEFAULT_SYSTEM
from oppods.data import ChannelMemmap, deterministic_split_indices
from oppods.metrics import per_sample_score, summarize_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the competition-constrained analytic baseline")
    parser.add_argument("--data", type=Path, default=Path("ziliao/data_train/H_train.npz"))
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1176)
    parser.add_argument("--group-size", type=int, default=24)
    parser.add_argument("--bits-per-symbol", type=int, default=8)
    parser.add_argument("--decoded-layers", type=int)
    parser.add_argument("--interference-scale", type=float, default=1.0)
    parser.add_argument("--no-agc", action="store_true")
    parser.add_argument("--decision-directed-iterations", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    data = ChannelMemmap(args.data)
    indices = deterministic_split_indices(len(data), seed=args.seed)["validation"][: args.samples]
    rng = np.random.default_rng(args.seed)
    bits_np = rng.integers(
        0,
        2,
        size=(args.samples, DEFAULT_SYSTEM.num_ue, DEFAULT_SYSTEM.max_bits_per_ue),
        dtype=np.int8,
    )
    snr_np = rng.uniform(
        DEFAULT_SYSTEM.snr_dl_min_db,
        DEFAULT_SYSTEM.snr_dl_max_db,
        size=(args.samples, DEFAULT_SYSTEM.num_ue),
    ).astype(np.float32)
    link = AnalyticMUMIMOLink(
        group_size=args.group_size,
        bits_per_symbol=args.bits_per_symbol,
        decoded_layers=args.decoded_layers,
        interference_scale=args.interference_scale,
        use_agc=not args.no_agc,
        decision_directed_iterations=args.decision_directed_iterations,
    ).to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    all_scores: list[np.ndarray] = []
    layer_score_batches: dict[int, list[np.ndarray]] = {layer: [] for layer in range(1, args.bits_per_symbol + 1)}
    started = time.perf_counter()
    for start in range(0, args.samples, args.batch_size):
        end = min(start + args.batch_size, args.samples)
        channel = torch.from_numpy(data.read(indices[start:end])).to(device)
        bits = torch.from_numpy(bits_np[start:end]).to(device=device, dtype=torch.float32)
        snr = torch.from_numpy(snr_np[start:end]).to(device)
        llr = link(channel, bits, snr, generator=generator)
        scores = torch.stack(
            [per_sample_score(bits[:, user], llr[:, user]) for user in range(DEFAULT_SYSTEM.num_ue)], dim=1
        )
        all_scores.append(scores.cpu().numpy())
        if llr.shape[-1] == DEFAULT_SYSTEM.num_downlink_subcarriers * args.bits_per_symbol:
            for layer, batches in layer_score_batches.items():
                prefix = DEFAULT_SYSTEM.num_downlink_subcarriers * layer
                prefix_scores = torch.stack(
                    [per_sample_score(bits[:, user], llr[:, user, :prefix]) for user in range(DEFAULT_SYSTEM.num_ue)],
                    dim=1,
                )
                batches.append(prefix_scores.cpu().numpy())

    values = np.concatenate(all_scores, axis=0)
    summary = summarize_scores(values)
    result = {
        "samples": args.samples,
        "group_size": args.group_size,
        "bits_per_symbol": args.bits_per_symbol,
        "decoded_layers": args.decoded_layers or args.bits_per_symbol,
        "interference_scale": args.interference_scale,
        "use_agc": not args.no_agc,
        "decision_directed_iterations": args.decision_directed_iterations,
        "efficiency": summary.efficiency,
        "fairness": summary.fairness,
        "final": summary.final,
        "user0_mean": float(values[:, 0].mean()),
        "user1_mean": float(values[:, 1].mean()),
        "elapsed_seconds": time.perf_counter() - started,
        "snr_slices": {},
    }
    if all(layer_score_batches.values()):
        layer_scores = {layer: np.concatenate(batches, axis=0) for layer, batches in layer_score_batches.items()}
        best_rule: dict[str, float | int] | None = None
        full_scores = layer_scores[args.bits_per_symbol]
        for low_layers in range(1, args.bits_per_symbol):
            for threshold in np.arange(-20.0, 10.01, 1.0):
                dynamic_scores = np.where(snr_np < threshold, layer_scores[low_layers], full_scores)
                dynamic_summary = summarize_scores(dynamic_scores)
                candidate: dict[str, float | int] = {
                    "low_layers": low_layers,
                    "threshold_db": float(threshold),
                    "efficiency": dynamic_summary.efficiency,
                    "fairness": dynamic_summary.fairness,
                    "final": dynamic_summary.final,
                }
                if best_rule is None or candidate["final"] > best_rule["final"]:
                    best_rule = candidate
        result["best_single_threshold_rule"] = best_rule
    for low in range(-20, 20, 5):
        mask = (snr_np >= low) & (snr_np < low + 5)
        selected = values[mask]
        result["snr_slices"][f"{low}:{low + 5}"] = {
            "count": int(selected.size),
            "mean": float(selected.mean()),
            "p10": float(np.percentile(selected, 10)),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    output = Path("artifacts/analytic_baseline_results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        output.with_suffix(".npz"),
        scores=values.astype(np.float32),
        snr_db=snr_np,
        indices=indices,
    )


if __name__ == "__main__":
    main()

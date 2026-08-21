from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from oppods.constants import DEFAULT_SYSTEM
from oppods.data import ChannelMemmap, deterministic_split_indices
from oppods.learned_baseline import LearnedFeedbackMUMIMOLink
from oppods.metrics import per_sample_score, summarize_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pretrained task feedback with the analytic link")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("ziliao/data_train/H_train.npz"))
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1176)
    parser.add_argument("--group-size", type=int, default=12)
    parser.add_argument("--bits-per-symbol", type=int, default=8)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--decision-directed-iterations", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    data = ChannelMemmap(args.data)
    indices = deterministic_split_indices(len(data), seed=args.seed)["validation"][: args.samples]
    rng = np.random.default_rng(args.seed)
    bits_np = rng.integers(0, 2, (args.samples, DEFAULT_SYSTEM.num_ue, DEFAULT_SYSTEM.max_bits_per_ue), dtype=np.int8)
    snr_np = rng.uniform(-20.0, 20.0, (args.samples, DEFAULT_SYSTEM.num_ue)).astype(np.float32)
    link = LearnedFeedbackMUMIMOLink(
        args.group_size,
        args.bits_per_symbol,
        args.width,
        args.layers,
        args.heads,
        decision_directed_iterations=args.decision_directed_iterations,
    ).to(device)
    checkpoint = link.load_feedback_checkpoint(args.checkpoint)
    link.eval()
    generator = torch.Generator(device=device).manual_seed(args.seed)

    score_batches: list[np.ndarray] = []
    layer_batches: dict[int, list[np.ndarray]] = {layer: [] for layer in range(1, args.bits_per_symbol + 1)}
    started = time.perf_counter()
    for start in range(0, args.samples, args.batch_size):
        end = min(start + args.batch_size, args.samples)
        channel = torch.from_numpy(data.read(indices[start:end])).to(device)
        bits = torch.from_numpy(bits_np[start:end]).to(device=device, dtype=torch.float32)
        snr = torch.from_numpy(snr_np[start:end]).to(device)
        llr = link(channel, bits, snr, generator=generator)
        for layer, batches in layer_batches.items():
            prefix = DEFAULT_SYSTEM.num_downlink_subcarriers * layer
            scores = torch.stack(
                [per_sample_score(bits[:, user], llr[:, user, :prefix]) for user in range(DEFAULT_SYSTEM.num_ue)],
                dim=1,
            )
            batches.append(scores.cpu().numpy())
        score_batches.append(layer_batches[args.bits_per_symbol][-1])

    values = np.concatenate(score_batches)
    summary = summarize_scores(values)
    layer_scores = {layer: np.concatenate(batches) for layer, batches in layer_batches.items()}
    best_rule: dict[str, float | int] | None = None
    for low_layers in range(1, args.bits_per_symbol):
        for threshold in np.arange(-20.0, 10.01, 1.0):
            dynamic = np.where(snr_np < threshold, layer_scores[low_layers], values)
            candidate_summary = summarize_scores(dynamic)
            candidate: dict[str, float | int] = {
                "low_layers": low_layers,
                "threshold_db": float(threshold),
                "efficiency": candidate_summary.efficiency,
                "fairness": candidate_summary.fairness,
                "final": candidate_summary.final,
            }
            if best_rule is None or candidate["final"] > best_rule["final"]:
                best_rule = candidate

    result = {
        "samples": args.samples,
        "checkpoint_step": checkpoint["step"],
        "checkpoint_validation_loss": checkpoint["best_validation"],
        "efficiency": summary.efficiency,
        "fairness": summary.fairness,
        "final": summary.final,
        "best_single_threshold_rule": best_rule,
        "elapsed_seconds": time.perf_counter() - started,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    output = Path("artifacts/learned_baseline_results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

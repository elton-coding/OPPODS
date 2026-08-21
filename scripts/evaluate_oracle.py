from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from oppods.constants import DEFAULT_SYSTEM
from oppods.data import ChannelMemmap, deterministic_split_indices
from oppods.metrics import per_sample_score, summarize_scores
from oppods.oracle import simulate_perfect_csi_rzf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate perfect-CSI RZF/LMMSE upper bounds")
    parser.add_argument("--data", type=Path, default=Path("ziliao/data_train/H_train.npz"))
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1176)
    parser.add_argument("--group-size", type=int, default=12)
    parser.add_argument("--modes", type=int, nargs="+", default=[1, 2, 4, 6, 8])
    parser.add_argument("--regularization-scale", type=float, default=1.0)
    parser.add_argument("--fairness-exponent", type=float)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    data = ChannelMemmap(args.data)
    split = deterministic_split_indices(len(data), seed=args.seed)["validation"]
    indices = split[: args.samples]
    rng = np.random.default_rng(args.seed)
    all_bits = rng.integers(
        0,
        2,
        size=(args.samples, DEFAULT_SYSTEM.num_ue, DEFAULT_SYSTEM.max_bits_per_ue),
        dtype=np.int8,
    )
    all_snr = rng.uniform(
        DEFAULT_SYSTEM.snr_dl_min_db,
        DEFAULT_SYSTEM.snr_dl_max_db,
        size=(args.samples, DEFAULT_SYSTEM.num_ue),
    ).astype(np.float32)

    results: dict[str, dict[str, float | int]] = {}
    started = time.perf_counter()
    for mode in args.modes:
        scores: list[np.ndarray] = []
        mode_started = time.perf_counter()
        generator = torch.Generator(device=device).manual_seed(args.seed + mode)
        for start in range(0, args.samples, args.batch_size):
            end = min(start + args.batch_size, args.samples)
            h = torch.from_numpy(data.read(indices[start:end])).to(device=device)
            bits = torch.from_numpy(all_bits[start:end]).to(device=device, dtype=torch.float32)
            snr = torch.from_numpy(all_snr[start:end]).to(device=device)
            llr = simulate_perfect_csi_rzf(
                h,
                bits,
                snr,
                bits_per_symbol=mode,
                group_size=args.group_size,
                regularization_scale=args.regularization_scale,
                fairness_exponent=args.fairness_exponent,
                generator=generator,
            )
            batch_scores = []
            for user in range(DEFAULT_SYSTEM.num_ue):
                batch_scores.append(per_sample_score(bits[:, user], llr[:, user]).cpu().numpy())
            scores.append(np.stack(batch_scores, axis=1))

        values = np.concatenate(scores, axis=0)
        summary = summarize_scores(values)
        elapsed = time.perf_counter() - mode_started
        result = {
            "bits_per_symbol": mode,
            "transmitted_bits_per_ue": DEFAULT_SYSTEM.num_downlink_subcarriers * mode,
            "efficiency": summary.efficiency,
            "fairness": summary.fairness,
            "final": summary.final,
            "user0_mean": float(values[:, 0].mean()),
            "user1_mean": float(values[:, 1].mean()),
            "elapsed_seconds": elapsed,
            "regularization_scale": args.regularization_scale,
            "fairness_exponent": args.fairness_exponent,
        }
        results[str(mode)] = result
        print(json.dumps(result, ensure_ascii=False))

    metadata = {
        "device": str(device),
        "torch": torch.__version__,
        "samples": args.samples,
        "group_size": args.group_size,
        "total_elapsed_seconds": time.perf_counter() - started,
        "results": results,
    }
    output = Path("artifacts/oracle_results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {output.resolve()}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import torch

from oppods.data import ChannelMemmap, deterministic_split_indices
from oppods.metrics import summarize_scores

PREFIXES = (1, 132, 264, 396, 528, 660, 792, 924, 1056)


def load_model_design(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("submission_model_design_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact batch-size-one submission evaluator")
    parser.add_argument("--submission", type=Path, default=Path("modelSubmit"))
    parser.add_argument("--data", type=Path, default=Path("ziliao/data_train/H_train.npz"))
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1176)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--search-prefixes", action="store_true")
    parser.add_argument("--prefix-scores-out", type=Path)
    parser.add_argument("--scores-out", type=Path, help="Save exact per-UE scores, SNRs, and output lengths")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    module = load_model_design(args.submission / "modelDesign.py")
    if args.threshold is not None:
        module.LOW_SNR_THRESHOLD_DB = args.threshold
    encoder = module.Encoder().to(device)
    transmitter = module.Transmitter().to(device)
    receiver = module.Receiver().to(device)
    encoder.load_state_dict(torch.load(args.submission / "encoder.pth", map_location=device, weights_only=True))
    transmitter.load_state_dict(torch.load(args.submission / "transmitter.pth", map_location=device, weights_only=True))
    receiver.load_state_dict(torch.load(args.submission / "receiver.pth", map_location=device, weights_only=True))
    encoder.eval()
    transmitter.eval()
    receiver.eval()

    data = ChannelMemmap(args.data)
    indices = deterministic_split_indices(len(data), seed=args.seed)["test"][: args.samples]
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    scores: list[float] = []
    lengths: list[int] = []
    prefix_scores: dict[int, list[float]] = {prefix: [] for prefix in PREFIXES}
    user_snrs: list[float] = []
    started = time.perf_counter()
    with torch.no_grad():
        for sample_index, data_index in enumerate(indices):
            channel = torch.from_numpy(data.read(int(data_index))).unsqueeze(0).to(device)
            bits_list = [torch.randint(0, 2, (1, 1152), dtype=torch.float32, device=device) for _ in range(2)]
            snr_dl = torch.from_numpy(rng.uniform(-20.0, 20.0, (2, 1)).astype(np.float32)).to(device)
            feedback_list = []
            for user in range(2):
                feedback = encoder(channel[:, user], snr_dl[user])
                assert feedback.shape == (1, 96) and feedback.dtype == torch.complex64
                feedback = feedback / torch.sqrt(torch.mean(torch.abs(feedback).square(), dim=1, keepdim=True))
                noise = torch.complex(
                    torch.randn_like(feedback.real) / math.sqrt(2.0),
                    torch.randn_like(feedback.real) / math.sqrt(2.0),
                )
                feedback_list.append(feedback + noise * torch.sqrt(torch.pow(10.0, -(snr_dl[user] - 10.0) / 10.0)))

            signal, ctrl = transmitter(bits_list, feedback_list, snr_dl)
            assert signal.shape == (1, 16, 144) and signal.dtype == torch.complex64
            assert ctrl.shape == (1, 5) and torch.all(ctrl == ctrl.square())
            energy = torch.mean(torch.sum(torch.abs(signal).square(), dim=1, keepdim=True), dim=(1, 2), keepdim=True)
            signal = signal / torch.sqrt(energy)
            for user in range(2):
                received = torch.sum(channel[:, user] * signal.unsqueeze(1), dim=2)
                noise = torch.complex(
                    torch.randn_like(received.real) / math.sqrt(2.0),
                    torch.randn_like(received.real) / math.sqrt(2.0),
                )
                received = received + noise * torch.sqrt(torch.pow(10.0, -snr_dl[user] / 10.0))[:, None, None]
                llr = receiver(received, channel[:, user], ctrl, snr_dl[user])
                assert llr.ndim == 2 and 0 < llr.shape[1] <= 1152
                assert llr.dtype == torch.float32 and torch.isfinite(llr).all()
                correct = ((llr >= 0) == (bits_list[user][:, : llr.shape[1]] >= 0.5)).sum()
                score = 100.0 * (float(correct) + 0.5 * (1152 - llr.shape[1])) / 1152
                scores.append(score)
                lengths.append(llr.shape[1])
                user_snrs.append(float(snr_dl[user].item()))
                if args.search_prefixes:
                    if llr.shape[1] < PREFIXES[-1]:
                        raise RuntimeError("prefix search requires full 1056-bit receiver output")
                    user_bits = bits_list[user]
                    for prefix in PREFIXES:
                        prefix_correct = (
                            (llr[:, :prefix] >= 0) == (user_bits[:, :prefix] >= 0.5)
                        ).sum()
                        prefix_scores[prefix].append(
                            100.0 * (float(prefix_correct) + 0.5 * (1152 - prefix)) / 1152
                        )
            if args.progress_every and (sample_index + 1) % args.progress_every == 0:
                print(
                    json.dumps(
                        {
                            "processed": sample_index + 1,
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    summary = summarize_scores(np.asarray(scores))
    if args.scores_out is not None:
        args.scores_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.scores_out,
            score=np.asarray(scores, dtype=np.float32),
            snr=np.asarray(user_snrs, dtype=np.float32),
            length=np.asarray(lengths, dtype=np.int16),
            data_index=np.repeat(indices.astype(np.int64), 2),
        )
    result: dict[str, object] = {
                "samples": len(indices),
                "scores": len(scores),
                "efficiency": summary.efficiency,
                "fairness": summary.fairness,
                "final": summary.final,
                "elapsed_seconds": time.perf_counter() - started,
                "short_outputs": int(np.sum(np.asarray(lengths) < 1152)),
                "min_output_length": min(lengths),
                "max_output_length": max(lengths),
            }
    if args.search_prefixes:
        snrs = np.asarray(user_snrs)
        arrays = {prefix: np.asarray(values) for prefix, values in prefix_scores.items()}
        if args.prefix_scores_out is not None:
            args.prefix_scores_out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.prefix_scores_out,
                snr=snrs,
                **{f"score_{prefix}": values for prefix, values in arrays.items()},
            )
        candidates = []
        for low_threshold in np.arange(-20.0, -4.99, 0.5):
            for middle_prefix in PREFIXES[1:-1]:
                for high_threshold in np.arange(low_threshold + 0.5, 20.01, 0.5):
                    selected = np.where(
                        snrs < low_threshold,
                        arrays[1],
                        np.where(snrs < high_threshold, arrays[middle_prefix], arrays[1056]),
                    )
                    candidate_summary = summarize_scores(selected)
                    candidates.append(
                        {
                            "low_threshold_db": float(low_threshold),
                            "middle_prefix": middle_prefix,
                            "high_threshold_db": float(high_threshold),
                            "efficiency": candidate_summary.efficiency,
                            "fairness": candidate_summary.fairness,
                            "final": candidate_summary.final,
                        }
                    )
        result["prefix_search"] = {
            "current_rule": summarize_scores(
                np.where(snrs < -15.5, arrays[1], arrays[1056])
            ).__dict__,
            "proposed_rule": summarize_scores(
                np.where(
                    snrs < -16.0,
                    arrays[1],
                    np.where(snrs < -9.5, arrays[924], arrays[1056]),
                )
            ).__dict__,
            "best": max(candidates, key=lambda item: item["final"]),
            "top": sorted(candidates, key=lambda item: item["final"], reverse=True)[:20],
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

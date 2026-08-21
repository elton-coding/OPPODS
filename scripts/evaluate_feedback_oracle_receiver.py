from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from oppods.channel import add_awgn, apply_downlink, complex_standard_normal, normalize_downlink, normalize_feedback
from oppods.data import ChannelMemmap, deterministic_split_indices
from oppods.denoised_link import DenoisedSparseMUMIMOLink
from oppods.metrics import per_sample_score, summarize_scores
from oppods.modulation import layered_qam_maxlog_llr, layered_qam_modulate
from oppods.oracle import _lmmse_observation, rzf_precoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure the receiver-observability upper bound for learned feedback")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/sparse_denoiser_task.pt"))
    parser.add_argument("--data", type=Path, default=Path("ziliao/data_train/H_train.npz"))
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1176)
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
    link = DenoisedSparseMUMIMOLink().to(device)
    link.load_denoiser_checkpoint(args.checkpoint)
    link.eval()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    score_batches = []
    for start in range(0, count, args.batch_size):
        end = min(start + args.batch_size, count)
        channel = torch.from_numpy(data.read(indices[start:end])).to(device)
        bits = torch.from_numpy(bits_np[start:end]).to(device=device, dtype=torch.float32)
        snr = torch.from_numpy(snr_np[start:end]).to(device)
        decoded_users = []
        for user in range(2):
            feedback = normalize_feedback(link.encoder(channel[:, user], snr[:, user]))
            noise = complex_standard_normal(tuple(feedback.shape), device=device, generator=generator)
            ul_variance = torch.pow(10.0, -(snr[:, user] - 10.0) / 10.0)
            decoded_users.append(
                link.transmitter.decoder(feedback + noise * torch.sqrt(ul_variance)[:, None], snr[:, user])
            )
        decoded = torch.stack(decoded_users, dim=2)
        precoder_group = rzf_precoder(decoded, torch.pow(10.0, -snr / 10.0), 1.5)
        precoder_sc = precoder_group.repeat_interleave(12, dim=1)
        symbols = torch.stack([layered_qam_modulate(bits[:, user], 8, central_boost=-0.5) for user in range(2)], dim=1)
        signal = torch.einsum("bstu,bus->bts", precoder_sc, symbols)
        signal, scale = normalize_downlink(signal)
        received, noise_variance = add_awgn(apply_downlink(channel, signal), snr, generator=generator)
        effective_matrix = torch.einsum("burts,bstv->bursv", channel, precoder_sc)
        effective_matrix = effective_matrix / scale[:, None, None, None, None]
        observation, gain, residual = _lmmse_observation(received, effective_matrix, noise_variance)
        llr = layered_qam_maxlog_llr(observation, gain, residual, 8, central_boost=-0.5)
        score_batches.append(
            torch.stack([per_sample_score(bits[:, user], llr[:, user]) for user in range(2)], dim=1).cpu().numpy()
        )
    scores = np.concatenate(score_batches)
    summary = summarize_scores(scores)
    print(
        json.dumps(
            {
                "samples": count,
                "efficiency": summary.efficiency,
                "fairness": summary.fairness,
                "final": summary.final,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

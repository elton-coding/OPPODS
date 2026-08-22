from __future__ import annotations

import argparse
import importlib.util
import json
import random
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from oppods.channel import complex_standard_normal, normalize_feedback
from oppods.data import ChannelMemmap, deterministic_split_indices
from oppods.metrics import lower_cvar, soft_per_sample_score


def load_model_design(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("score_aligned_model_design", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the feedback denoiser through soft official bit score")
    parser.add_argument("--model-design", type=Path, default=Path("modelSubmit/modelDesign.py"))
    parser.add_argument("--data", type=Path, default=Path("ziliao/data_train/H_train.npz"))
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--score-tail-weight", type=float, default=0.3)
    parser.add_argument("--score-tail-alpha", type=float, default=0.1)
    parser.add_argument("--llr-temperature", type=float, default=8.0)
    parser.add_argument("--anchor-weight", type=float, default=1e-3)
    parser.add_argument("--weak-snr-min", type=float, default=-15.5)
    parser.add_argument("--weak-snr-max", type=float, default=-9.5)
    parser.add_argument("--strong-snr-min", type=float, default=-9.5)
    parser.add_argument("--strong-snr-max", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1176)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("steps and batch size must be positive")
    if not 0.0 <= args.score_tail_weight <= 1.0:
        raise ValueError("--score-tail-weight must be in [0, 1]")
    if not 0.0 < args.score_tail_alpha <= 1.0:
        raise ValueError("--score-tail-alpha must be in (0, 1]")
    if args.llr_temperature <= 0.0:
        raise ValueError("--llr-temperature must be positive")
    if args.weak_snr_min >= args.weak_snr_max or args.strong_snr_min >= args.strong_snr_max:
        raise ValueError("each SNR interval must have increasing bounds")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def anchor_penalty(
    parameters: dict[str, torch.Tensor], anchors: dict[str, torch.Tensor]
) -> torch.Tensor:
    penalties = [(value - anchors[name]).square().mean() for name, value in parameters.items()]
    return torch.stack(penalties).mean()


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = load_model_design(args.model_design)
    encoder = module.Encoder().to(device).eval()
    transmitter = module.Transmitter().to(device)
    receiver = module.Receiver().to(device).eval()
    checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
    transmitter.decoder.load_state_dict(checkpoint["denoiser"])
    decoder_parameters = dict(transmitter.decoder.named_parameters())
    anchors = {name: value.detach().clone() for name, value in decoder_parameters.items()}
    optimizer = torch.optim.AdamW(
        decoder_parameters.values(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    data = ChannelMemmap(args.data)
    train_indices = deterministic_split_indices(len(data), seed=args.seed)["train"]
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {
        "model_design": str(args.model_design),
        "data": str(args.data),
        "resume": str(args.resume),
        "output": str(args.output),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)

    started = time.perf_counter()
    last_objective = torch.tensor(0.0)
    for step in range(args.steps):
        indices = rng.choice(train_indices, args.batch_size, replace=False)
        channel = torch.from_numpy(data.read(indices)).to(device)
        bits_list = [
            torch.randint(0, 2, (args.batch_size, 1152), device=device, generator=generator).to(torch.float32)
            for _ in range(2)
        ]
        weak_snr = torch.empty(args.batch_size, device=device).uniform_(
            args.weak_snr_min, args.weak_snr_max, generator=generator
        )
        strong_snr = torch.empty(args.batch_size, device=device).uniform_(
            args.strong_snr_min, args.strong_snr_max, generator=generator
        )
        weak_user = int(torch.randint(0, 2, (), device=device, generator=generator))
        snr_dl = [strong_snr, strong_snr]
        snr_dl[weak_user] = weak_snr
        snr_dl = torch.stack(snr_dl)

        feedback_list: list[torch.Tensor] = []
        with torch.no_grad():
            for user in range(2):
                feedback = normalize_feedback(encoder(channel[:, user], snr_dl[user]))
                feedback_noise = complex_standard_normal(
                    tuple(feedback.shape), device=device, generator=generator
                )
                feedback_variance = torch.pow(10.0, -(snr_dl[user] - 10.0) / 10.0)
                feedback_list.append(feedback + feedback_noise * torch.sqrt(feedback_variance)[:, None])

        signal, ctrl = transmitter(bits_list, feedback_list, snr_dl)
        energy = torch.mean(torch.sum(torch.abs(signal).square(), dim=1), dim=1).clamp_min(1e-9)
        signal = signal / torch.sqrt(energy)[:, None, None]
        soft_scores = []
        for user in range(2):
            received = torch.sum(channel[:, user] * signal.unsqueeze(1), dim=2)
            downlink_noise = complex_standard_normal(tuple(received.shape), device=device, generator=generator)
            received = received + downlink_noise * torch.sqrt(torch.pow(10.0, -snr_dl[user] / 10.0))[:, None, None]
            llr = receiver(received, channel[:, user], ctrl, snr_dl[user])
            soft_scores.append(
                soft_per_sample_score(bits_list[user], llr / args.llr_temperature)
            )
        scores = torch.stack(soft_scores, dim=1)
        mean_score = scores.mean()
        tail_score = lower_cvar(scores.reshape(-1), alpha=args.score_tail_alpha)
        objective = (1.0 - args.score_tail_weight) * mean_score + args.score_tail_weight * tail_score
        regularization = anchor_penalty(decoder_parameters, anchors)
        loss = -objective / 100.0 + args.anchor_weight * regularization

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = clip_grad_norm_(decoder_parameters.values(), 5.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError("non-finite score-aligned gradient")
        optimizer.step()
        last_objective = objective.detach()
        if step % args.log_every == 0 or step + 1 == args.steps:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": float(loss.detach()),
                        "mean_soft_score": float(mean_score.detach()),
                        "tail_soft_score": float(tail_score.detach()),
                        "objective": float(objective.detach()),
                        "anchor_penalty": float(regularization.detach()),
                        "grad_norm": float(grad_norm),
                        "steps_per_second": (step + 1) / max(time.perf_counter() - started, 1e-6),
                    }
                ),
                flush=True,
            )

    torch.save(
        {
            "step": args.steps - 1,
            "best_validation": -float(last_objective),
            "denoiser": transmitter.decoder.state_dict(),
            "config": config,
        },
        args.output,
    )
    print(f"saved {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()

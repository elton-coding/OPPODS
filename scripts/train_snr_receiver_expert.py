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
    spec = importlib.util.spec_from_file_location("snr_receiver_expert_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one score-aligned SNR Receiver LLR expert")
    parser.add_argument("--submission", type=Path, default=Path("artifacts/candidates/snr_expert_clone12"))
    parser.add_argument("--data", type=Path, default=Path("ziliao/data_train/H_train.npz"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expert-index", type=int, required=True)
    parser.add_argument("--snr-min", type=float, required=True)
    parser.add_argument("--snr-max", type=float, required=True)
    parser.add_argument("--prefix-bits", type=int, choices=(924, 1056), required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--score-tail-weight", type=float, default=0.3)
    parser.add_argument("--score-tail-alpha", type=float, default=0.2)
    parser.add_argument("--llr-temperature", type=float, default=4.0)
    parser.add_argument("--correction-weight", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1176)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--validation-batches", type=int, default=16)
    return parser.parse_args()


def validate_args(args: argparse.Namespace, expert_count: int) -> None:
    if not 0 <= args.expert_index < expert_count:
        raise ValueError(f"--expert-index must be in [0, {expert_count - 1}]")
    if not -20.0 <= args.snr_min < args.snr_max <= 20.0:
        raise ValueError("SNR interval must be increasing and inside [-20, 20]")
    if args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("steps and batch size must be positive")
    if not 0.0 <= args.score_tail_weight <= 1.0:
        raise ValueError("--score-tail-weight must be in [0, 1]")
    if not 0.0 < args.score_tail_alpha <= 1.0:
        raise ValueError("--score-tail-alpha must be in (0, 1]")
    if args.llr_temperature <= 0.0:
        raise ValueError("--llr-temperature must be positive")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_submission(args: argparse.Namespace, device: torch.device) -> tuple[ModuleType, torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    module = load_model_design(Path("modelSubmit/modelDesign.py"))
    encoder = module.Encoder().to(device).eval()
    transmitter = module.Transmitter().to(device).eval()
    receiver = module.Receiver().to(device).eval()
    encoder.load_state_dict(torch.load(args.submission / "encoder.pth", map_location=device, weights_only=True))
    transmitter.load_state_dict(
        torch.load(args.submission / "transmitter.pth", map_location=device, weights_only=True)
    )
    receiver_state = torch.load(args.submission / "receiver.pth", map_location=device, weights_only=True)
    receiver.load_state_dict(receiver_state, strict=False)
    return module, encoder, transmitter, receiver


@torch.no_grad()
def simulate_raw_llr(
    channel: torch.Tensor,
    target_snr: torch.Tensor,
    partner_snr: torch.Tensor,
    target_user: int,
    bits_list: list[torch.Tensor],
    encoder: torch.nn.Module,
    transmitter: torch.nn.Module,
    receiver: torch.nn.Module,
    generator: torch.Generator,
) -> torch.Tensor:
    batch = channel.shape[0]
    snr_dl = [partner_snr, partner_snr]
    snr_dl[target_user] = target_snr
    snr_dl = torch.stack(snr_dl)
    feedback_list = []
    for user in range(2):
        feedback = normalize_feedback(encoder(channel[:, user], snr_dl[user]))
        feedback_noise = complex_standard_normal(tuple(feedback.shape), device=channel.device, generator=generator)
        feedback_variance = torch.pow(10.0, -(snr_dl[user] - 10.0) / 10.0)
        feedback_list.append(feedback + feedback_noise * torch.sqrt(feedback_variance)[:, None])
    signal, ctrl = transmitter(bits_list, feedback_list, snr_dl)
    energy = torch.mean(torch.sum(torch.abs(signal).square(), dim=1), dim=1).clamp_min(1e-9)
    signal = signal / torch.sqrt(energy)[:, None, None]
    received = torch.sum(channel[:, target_user] * signal.unsqueeze(1), dim=2)
    downlink_noise = complex_standard_normal(tuple(received.shape), device=channel.device, generator=generator)
    received = received + downlink_noise * torch.sqrt(torch.pow(10.0, -target_snr / 10.0))[:, None, None]
    receiver.disable_llr_experts = True
    receiver.training_prefix_bits = 1056
    raw_llr = receiver(received, channel[:, target_user], ctrl, target_snr)
    if raw_llr.shape != (batch, 1056):
        raise RuntimeError(f"expected raw LLR shape {(batch, 1056)}, got {tuple(raw_llr.shape)}")
    return raw_llr


def draw_batch(
    data: ChannelMemmap,
    indices: np.ndarray,
    rng: np.random.Generator,
    generator: torch.Generator,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, list[torch.Tensor]]:
    selected = rng.choice(indices, args.batch_size, replace=False)
    channel = torch.from_numpy(data.read(selected)).to(device)
    target_snr = torch.empty(args.batch_size, device=device).uniform_(
        args.snr_min, args.snr_max, generator=generator
    )
    partner_snr = torch.empty(args.batch_size, device=device).uniform_(-20.0, 20.0, generator=generator)
    target_user = int(torch.randint(0, 2, (), device=device, generator=generator))
    bits_list = [
        torch.randint(0, 2, (args.batch_size, 1152), device=device, generator=generator).to(torch.float32)
        for _ in range(2)
    ]
    return channel, target_snr, partner_snr, target_user, bits_list


def score_objective(
    expert: torch.nn.Module,
    raw_llr: torch.Tensor,
    snr: torch.Tensor,
    bits: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    corrected = expert(raw_llr, snr)[:, : args.prefix_bits]
    scores = soft_per_sample_score(bits, corrected / args.llr_temperature)
    mean_score = scores.mean()
    tail_score = lower_cvar(scores, alpha=args.score_tail_alpha)
    objective = (1.0 - args.score_tail_weight) * mean_score + args.score_tail_weight * tail_score
    correction = corrected - raw_llr[:, : args.prefix_bits]
    correction_energy = correction.square().mean()
    loss = -objective / 100.0 + args.correction_weight * correction_energy
    return loss, {
        "mean_soft_score": mean_score.detach(),
        "tail_soft_score": tail_score.detach(),
        "objective": objective.detach(),
        "correction_rms": torch.sqrt(correction_energy.detach()),
    }


@torch.no_grad()
def validate(
    data: ChannelMemmap,
    indices: np.ndarray,
    encoder: torch.nn.Module,
    transmitter: torch.nn.Module,
    receiver: torch.nn.Module,
    expert: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    rng = np.random.default_rng(args.seed + 50000)
    generator = torch.Generator(device=device).manual_seed(args.seed + 50000)
    sums: dict[str, float] = {}
    expert.eval()
    for _ in range(args.validation_batches):
        channel, target_snr, partner_snr, target_user, bits_list = draw_batch(
            data, indices, rng, generator, args, device
        )
        raw_llr = simulate_raw_llr(
            channel, target_snr, partner_snr, target_user, bits_list, encoder, transmitter, receiver, generator
        )
        loss, metrics = score_objective(expert, raw_llr, target_snr, bits_list[target_user], args)
        metrics = {"loss": loss.detach()} | metrics
        for name, value in metrics.items():
            sums[name] = sums.get(name, 0.0) + float(value)
    expert.train()
    return {name: value / args.validation_batches for name, value in sums.items()}


def checkpoint_payload(
    transmitter: torch.nn.Module,
    receiver: torch.nn.Module,
    step: int,
    best_validation: float,
    config: dict[str, object],
) -> dict[str, object]:
    denoisers = [transmitter.decoder, *transmitter.expert_decoders]
    return {
        "step": step,
        "best_validation": best_validation,
        "denoiser": denoisers[0].state_dict(),
        "experts": [denoiser.state_dict() for denoiser in denoisers],
        "receiver": receiver.state_dict(),
        "config": config,
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, encoder, transmitter, receiver = load_submission(args, device)
    validate_args(args, len(receiver.llr_experts))
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in transmitter.parameters():
        parameter.requires_grad_(False)
    for parameter in receiver.parameters():
        parameter.requires_grad_(False)
    expert = receiver.llr_experts[args.expert_index]
    for parameter in expert.parameters():
        parameter.requires_grad_(True)
    expert.train()
    optimizer = torch.optim.AdamW(expert.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.learning_rate * 0.1)

    data = ChannelMemmap(args.data)
    splits = deterministic_split_indices(len(data), seed=args.seed)
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config: dict[str, object] = vars(args) | {
        "submission": str(args.submission),
        "data": str(args.data),
        "output": str(args.output),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)
    best_validation = float("inf")
    started = time.perf_counter()
    for step in range(args.steps):
        channel, target_snr, partner_snr, target_user, bits_list = draw_batch(
            data, splits["train"], rng, generator, args, device
        )
        raw_llr = simulate_raw_llr(
            channel, target_snr, partner_snr, target_user, bits_list, encoder, transmitter, receiver, generator
        )
        loss, metrics = score_objective(expert, raw_llr, target_snr, bits_list[target_user], args)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = clip_grad_norm_(expert.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        if step % args.log_every == 0:
            print(
                json.dumps(
                    {name: float(value) for name, value in metrics.items()}
                    | {
                        "step": step,
                        "loss": float(loss.detach()),
                        "grad_norm": float(grad_norm),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "steps_per_second": (step + 1) / max(time.perf_counter() - started, 1e-6),
                    }
                ),
                flush=True,
            )
        if (step + 1) % args.validate_every == 0 or step + 1 == args.steps:
            validation = validate(
                data,
                splits["validation"],
                encoder,
                transmitter,
                receiver,
                expert,
                args,
                device,
            )
            print(json.dumps({"step": step, "validation": validation}), flush=True)
            if validation["loss"] < best_validation:
                best_validation = validation["loss"]
                torch.save(
                    checkpoint_payload(transmitter, receiver, step, best_validation, config),
                    args.output,
                )
                print(f"saved {args.output.resolve()} validation_loss={best_validation:.6f}", flush=True)


if __name__ == "__main__":
    main()

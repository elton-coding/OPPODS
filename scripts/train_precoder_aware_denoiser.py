from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from oppods.analytic_baseline import _task_feedback
from oppods.channel import complex_standard_normal, normalize_feedback
from oppods.constants import DEFAULT_SYSTEM
from oppods.data import ChannelMemmap, deterministic_split_indices
from oppods.metrics import lower_cvar
from oppods.oracle import rzf_precoder
from oppods.sparse_denoiser import SparseFeedbackDenoiser, parameter_count
from oppods.sparse_feedback import SparseDelayEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune sparse denoising through the robust RZF task")
    parser.add_argument("--data", type=Path, default=Path("ziliao/data_train/H_train.npz"))
    parser.add_argument("--resume", type=Path, default=Path("checkpoints/sparse_denoiser_stage2.pt"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/sparse_denoiser_task.pt"))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--regularization-scale", type=float, default=1.5)
    parser.add_argument("--reconstruction-weight", type=float, default=0.05)
    parser.add_argument("--tail-weight", type=float, default=0.1)
    parser.add_argument("--snr-min", type=float, default=-20.0)
    parser.add_argument("--snr-max", type=float, default=20.0)
    parser.add_argument("--minimum-profile-max-snr", type=float)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1176)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--validation-samples", type=int, default=2048)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate_args(args: argparse.Namespace) -> None:
    if not -20.0 <= args.snr_min < args.snr_max <= 20.0:
        raise ValueError("--snr-min/--snr-max must define an increasing interval within [-20, 20]")


def forward_task(
    channel: torch.Tensor,
    snr_dl: torch.Tensor,
    encoder: SparseDelayEncoder,
    denoiser: SparseFeedbackDenoiser,
    regularization_scale: float,
    reconstruction_weight: float,
    tail_weight: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch = channel.shape[0]
    decoded_users = []
    clean_users = []
    coefficient_nmse_users = []
    true_users = []
    for user in range(DEFAULT_SYSTEM.num_ue):
        local_channel = channel[:, user]
        local_snr = snr_dl[:, user]
        true_effective = _task_feedback(local_channel, encoder.group_size)
        clean_feedback = normalize_feedback(encoder(local_channel, local_snr))
        clean_coefficients = clean_feedback[:, : encoder.mode_count * 16].reshape(batch, encoder.mode_count, 16)
        noise = complex_standard_normal(tuple(clean_feedback.shape), device=channel.device, generator=generator)
        noise_variance = torch.pow(10.0, -(local_snr - DEFAULT_SYSTEM.snr_ul_gap_db) / 10.0)
        prediction = denoiser.forward_details(clean_feedback + noise * torch.sqrt(noise_variance)[:, None], local_snr)
        coefficient_error = torch.sum(torch.abs(prediction.coefficients - clean_coefficients).square(), dim=(1, 2))
        coefficient_energy = torch.sum(torch.abs(clean_coefficients).square(), dim=(1, 2)).clamp_min(1e-9)
        coefficient_nmse_users.append(coefficient_error / coefficient_energy)
        decoded_users.append(prediction.effective_channel)
        clean_users.append(clean_coefficients)
        true_users.append(true_effective)

    decoded = torch.stack(decoded_users, dim=2)
    true_effective = torch.stack(true_users, dim=2)
    noise_variance_dl = torch.pow(10.0, -snr_dl / 10.0)
    precoder = rzf_precoder(decoded, noise_variance_dl, regularization_scale)
    effective_matrix = true_effective @ precoder
    desired_power = torch.abs(torch.diagonal(effective_matrix, dim1=-2, dim2=-1)).square()
    total_power = torch.sum(torch.abs(effective_matrix).square(), dim=-1)
    interference_power = (total_power - desired_power).clamp_min(0.0)
    sinr = desired_power / (interference_power + noise_variance_dl[:, None, :]).clamp_min(1e-9)
    utility_per_user = torch.log1p(sinr).mean(dim=1).reshape(-1)
    mean_utility = utility_per_user.mean()
    tail_utility = lower_cvar(utility_per_user, alpha=0.2)
    coefficient_nmse = torch.stack(coefficient_nmse_users, dim=1).mean()
    task_utility = (1.0 - tail_weight) * mean_utility + tail_weight * tail_utility
    loss = -task_utility + reconstruction_weight * coefficient_nmse
    metrics = {
        "loss": loss.detach(),
        "mean_utility": mean_utility.detach(),
        "tail_utility": tail_utility.detach(),
        "sinr_db": (10.0 * torch.log10(sinr.mean().clamp_min(1e-9))).detach(),
        "coefficient_nmse": coefficient_nmse.detach(),
    }
    return loss, metrics


@torch.no_grad()
def validate(
    data: ChannelMemmap,
    indices: np.ndarray,
    encoder: SparseDelayEncoder,
    denoiser: SparseFeedbackDenoiser,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    denoiser.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    rng = np.random.default_rng(seed)
    sums: dict[str, float] = {}
    count = 0
    for start in range(0, len(indices), args.batch_size):
        batch_indices = indices[start : start + args.batch_size]
        channel = torch.from_numpy(data.read(batch_indices)).to(device)
        snr = torch.from_numpy(
            rng.uniform(args.snr_min, args.snr_max, (len(batch_indices), 2)).astype(np.float32)
        ).to(device)
        if args.minimum_profile_max_snr is not None:
            mask = snr.max(dim=1).values < args.minimum_profile_max_snr
            replacement = torch.from_numpy(
                rng.uniform(args.minimum_profile_max_snr, 20.0, int(mask.sum())).astype(np.float32)
            ).to(device)
            snr[mask, 0] = replacement
        _, metrics = forward_task(
            channel,
            snr,
            encoder,
            denoiser,
            args.regularization_scale,
            args.reconstruction_weight,
            args.tail_weight,
            generator,
        )
        for name, value in metrics.items():
            sums[name] = sums.get(name, 0.0) + float(value) * len(batch_indices)
        count += len(batch_indices)
    denoiser.train()
    return {name: value / count for name, value in sums.items()}


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = ChannelMemmap(args.data)
    splits = deterministic_split_indices(len(data), seed=args.seed)
    validation_indices = splits["validation"][: args.validation_samples]
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    encoder = SparseDelayEncoder().to(device)
    denoiser = SparseFeedbackDenoiser(width=args.width, layers=args.layers, heads=args.heads).to(device)
    checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
    denoiser.load_state_dict(checkpoint["denoiser"])
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=5e-6)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "parameters": parameter_count(denoiser),
        "args": vars(args) | {"data": str(args.data), "resume": str(args.resume), "output": str(args.output)},
    }
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    best_validation = float("inf")
    started = time.perf_counter()
    for step in range(args.steps):
        indices = rng.choice(splits["train"], args.batch_size, replace=False)
        channel = torch.from_numpy(data.read(indices)).to(device)
        snr = torch.empty(args.batch_size, 2, device=device).uniform_(
            args.snr_min, args.snr_max, generator=generator
        )
        if args.minimum_profile_max_snr is not None:
            mask = snr.max(dim=1).values < args.minimum_profile_max_snr
            replacement = torch.empty(int(mask.sum()), device=device).uniform_(
                args.minimum_profile_max_snr, 20.0, generator=generator
            )
            snr[mask, 0] = replacement
        loss, metrics = forward_task(
            channel,
            snr,
            encoder,
            denoiser,
            args.regularization_scale,
            args.reconstruction_weight,
            args.tail_weight,
            generator,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = clip_grad_norm_(denoiser.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        if step % args.log_every == 0:
            values = {name: float(value) for name, value in metrics.items()}
            values.update(
                {
                    "step": step,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "grad_norm": float(grad_norm),
                    "steps_per_second": (step + 1) / max(time.perf_counter() - started, 1e-6),
                }
            )
            print(json.dumps(values, ensure_ascii=False))
        if (step + 1) % args.validate_every == 0 or step + 1 == args.steps:
            validation = validate(
                data,
                validation_indices,
                encoder,
                denoiser,
                args,
                device,
                args.seed + 50000,
            )
            print(json.dumps({"step": step, "validation": validation}, ensure_ascii=False))
            if validation["loss"] < best_validation:
                best_validation = validation["loss"]
                torch.save(
                    {
                        "step": step,
                        "best_validation": best_validation,
                        "denoiser": denoiser.state_dict(),
                        "config": metadata,
                    },
                    args.output,
                )
                print(f"saved {args.output.resolve()} validation_loss={best_validation:.6f}")


if __name__ == "__main__":
    main()

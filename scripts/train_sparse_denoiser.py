from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from oppods.analytic_baseline import _task_feedback
from oppods.channel import complex_standard_normal, normalize_feedback
from oppods.constants import DEFAULT_SYSTEM
from oppods.data import ChannelMemmap, deterministic_split_indices
from oppods.sparse_denoiser import SparseFeedbackDenoiser, parameter_count, sparse_denoising_loss
from oppods.sparse_feedback import SparseDelayEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train residual denoising for deterministic task feedback")
    parser.add_argument("--data", type=Path, default=Path("ziliao/data_train/H_train.npz"))
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--mode-count", type=int, default=6)
    parser.add_argument("--wiener-noise-scale", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=1176)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--validation-samples", type=int, default=2048)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/sparse_denoiser_best.pt"))
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_batch(
    data: ChannelMemmap,
    indices: np.ndarray,
    encoder: SparseDelayEncoder,
    denoiser: SparseFeedbackDenoiser,
    device: torch.device,
    generator: torch.Generator,
    snr_dl: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    channel = torch.from_numpy(data.read(indices)).to(device).reshape(-1, 2, 16, 144)
    target_effective = _task_feedback(channel, encoder.group_size)
    clean_feedback = normalize_feedback(encoder(channel, snr_dl))
    clean_coefficients = clean_feedback[:, : encoder.mode_count * 16].reshape(-1, encoder.mode_count, 16)
    noise = complex_standard_normal(tuple(clean_feedback.shape), device=device, generator=generator)
    noise_variance = torch.pow(10.0, -(snr_dl - DEFAULT_SYSTEM.snr_ul_gap_db) / 10.0)
    prediction = denoiser.forward_details(clean_feedback + noise * torch.sqrt(noise_variance)[:, None], snr_dl)
    return sparse_denoising_loss(prediction, clean_coefficients, target_effective, snr_dl)


@torch.no_grad()
def validate(
    data: ChannelMemmap,
    indices: np.ndarray,
    encoder: SparseDelayEncoder,
    denoiser: SparseFeedbackDenoiser,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    denoiser.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    sums: dict[str, float] = {}
    count = 0
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        local_count = 2 * len(batch_indices)
        snr_dl = torch.linspace(-20.0, 20.0, local_count, device=device)
        _, metrics = build_batch(data, batch_indices, encoder, denoiser, device, generator, snr_dl)
        for name, value in metrics.items():
            sums[name] = sums.get(name, 0.0) + float(value) * local_count
        count += local_count
    denoiser.train()
    return {name: value / count for name, value in sums.items()}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = ChannelMemmap(args.data)
    splits = deterministic_split_indices(len(data), seed=args.seed)
    validation_indices = splits["validation"][: args.validation_samples]
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    encoder = SparseDelayEncoder(mode_count=args.mode_count).to(device)
    denoiser = SparseFeedbackDenoiser(
        mode_count=args.mode_count,
        width=args.width,
        layers=args.layers,
        heads=args.heads,
        wiener_noise_scale=args.wiener_noise_scale,
    ).to(device)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        denoiser.load_state_dict(checkpoint["denoiser"])
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(Path("runs") / time.strftime("sparse_denoiser_%Y%m%d_%H%M%S"))
    metadata = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "parameters": parameter_count(denoiser),
        "args": vars(args) | {"data": str(args.data), "output": str(args.output), "resume": str(args.resume)},
    }
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    best_validation = float("inf")
    started = time.perf_counter()
    for step in range(args.steps):
        indices = rng.choice(splits["train"], args.batch_size, replace=False)
        snr_dl = torch.empty(2 * args.batch_size, device=device).uniform_(-20.0, 20.0, generator=generator)
        loss, metrics = build_batch(data, indices, encoder, denoiser, device, generator, snr_dl)
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
            for name, value in values.items():
                if name != "step":
                    writer.add_scalar(f"train/{name}", value, step)

        if (step + 1) % args.validate_every == 0 or step + 1 == args.steps:
            validation = validate(
                data,
                validation_indices,
                encoder,
                denoiser,
                args.batch_size,
                device,
                args.seed + step,
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
    writer.close()


if __name__ == "__main__":
    main()

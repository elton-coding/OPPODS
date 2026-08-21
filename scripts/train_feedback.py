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
from oppods.feedback_model import (
    TaskFeedbackDecoder,
    TaskOrientedEncoder,
    feedback_reconstruction_loss,
    parameter_count,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain task-oriented DJSCC feedback")
    parser.add_argument("--data", type=Path, default=Path("ziliao/data_train/H_train.npz"))
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--group-size", type=int, default=12)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1176)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--validation-samples", type=int, default=2048)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/feedback_best.pt"))
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def noisy_feedback(
    encoder: TaskOrientedEncoder,
    channel: torch.Tensor,
    snr_dl: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    signal = normalize_feedback(encoder(channel, snr_dl))
    snr_ul = snr_dl - DEFAULT_SYSTEM.snr_ul_gap_db
    noise = complex_standard_normal(tuple(signal.shape), device=signal.device, generator=generator)
    return signal + noise * torch.sqrt(torch.pow(10.0, -snr_ul / 10.0))[:, None]


@torch.no_grad()
def validate(
    encoder: TaskOrientedEncoder,
    decoder: TaskFeedbackDecoder,
    data: ChannelMemmap,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    encoder.eval()
    decoder.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    metric_sums: dict[str, float] = {}
    count = 0
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        channel = torch.from_numpy(data.read(batch_indices)).to(device).reshape(-1, 2, 16, 144)
        local_count = channel.shape[0]
        snr_dl = torch.linspace(-20.0, 20.0, local_count, device=device)
        target = _task_feedback(channel, encoder.group_size)
        feedback = noisy_feedback(encoder, channel, snr_dl, generator)
        prediction = decoder(feedback, snr_dl)
        _, metrics = feedback_reconstruction_loss(prediction, target, snr_dl)
        for name, value in metrics.items():
            metric_sums[name] = metric_sums.get(name, 0.0) + float(value) * local_count
        count += local_count
    encoder.train()
    decoder.train()
    return {name: value / count for name, value in metric_sums.items()}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = ChannelMemmap(args.data)
    splits = deterministic_split_indices(len(data), seed=args.seed)
    train_indices = splits["train"]
    validation_indices = splits["validation"][: args.validation_samples]
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    encoder = TaskOrientedEncoder(args.group_size, args.width, args.layers, args.heads).to(device)
    decoder = TaskFeedbackDecoder(args.group_size, args.width, args.layers, args.heads).to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-5)
    start_step = 0
    best_validation = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = int(checkpoint["step"]) + 1
        best_validation = float(checkpoint["best_validation"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_name = time.strftime("feedback_%Y%m%d_%H%M%S")
    writer = SummaryWriter(Path("runs") / run_name)
    metadata = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "encoder_parameters": parameter_count(encoder),
        "decoder_parameters": parameter_count(decoder),
        "args": vars(args) | {"data": str(args.data), "output": str(args.output), "resume": str(args.resume)},
    }
    print(json.dumps(metadata, ensure_ascii=False, indent=2))

    started = time.perf_counter()
    for step in range(start_step, args.steps):
        batch_indices = rng.choice(train_indices, args.batch_size, replace=False)
        channel = torch.from_numpy(data.read(batch_indices)).to(device).reshape(-1, 2, 16, 144)
        snr_dl = torch.empty(channel.shape[0], device=device).uniform_(-20.0, 20.0, generator=generator)
        target = _task_feedback(channel, args.group_size)
        feedback = noisy_feedback(encoder, channel, snr_dl, generator)
        prediction = decoder(feedback, snr_dl)
        loss, metrics = feedback_reconstruction_loss(prediction, target, snr_dl)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 5.0)
        optimizer.step()
        scheduler.step()

        if step % args.log_every == 0:
            elapsed = time.perf_counter() - started
            values = {name: float(value) for name, value in metrics.items()}
            values.update(
                {
                    "step": step,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "grad_norm": float(grad_norm),
                    "steps_per_second": (step - start_step + 1) / max(elapsed, 1e-6),
                }
            )
            print(json.dumps(values, ensure_ascii=False))
            for name, value in values.items():
                if name != "step":
                    writer.add_scalar(f"train/{name}", value, step)

        if (step + 1) % args.validate_every == 0 or step + 1 == args.steps:
            validation = validate(
                encoder,
                decoder,
                data,
                validation_indices,
                args.batch_size,
                device,
                args.seed + step,
            )
            print(json.dumps({"step": step, "validation": validation}, ensure_ascii=False))
            for name, value in validation.items():
                writer.add_scalar(f"validation/{name}", value, step)
            if validation["loss"] < best_validation:
                best_validation = validation["loss"]
                torch.save(
                    {
                        "step": step,
                        "best_validation": best_validation,
                        "encoder": encoder.state_dict(),
                        "decoder": decoder.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "config": metadata,
                    },
                    args.output,
                )
                print(f"saved {args.output.resolve()} validation_loss={best_validation:.6f}")
    writer.close()


if __name__ == "__main__":
    main()

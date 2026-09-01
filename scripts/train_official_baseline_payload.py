from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
from torch import nn

from oppods.data import ChannelMemmap, deterministic_split_indices
from oppods.metrics import per_sample_score, summarize_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the organizer neural baseline from scratch while changing only B and bits/RE"
    )
    parser.add_argument("--bits-per-re", type=int, choices=range(1, 9), required=True)
    parser.add_argument("--payload-bits", type=int, required=True)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validate-every", type=int, default=1_000)
    parser.add_argument("--validation-samples", type=int, default=1_000)
    parser.add_argument("--validation-batch-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1212)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data", type=Path, default=Path("ziliao/data_train/H_train.npz"))
    parser.add_argument(
        "--model-design",
        type=Path,
        default=Path("research/official_baseline_payload/modelDesign.py"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    capacity = 144 * args.bits_per_re
    if not 0 < args.payload_bits <= capacity:
        parser.error(f"--payload-bits must be in [1, {capacity}]")
    if args.steps <= 0 or args.batch_size <= 0:
        parser.error("--steps and --batch-size must be positive")
    return args


def load_model_design(path: Path, bits_per_re: int, payload_bits: int) -> ModuleType:
    spec = importlib.util.spec_from_file_location("official_baseline_payload_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import model design from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.NUM_BITS_PER_RE = bits_per_re
    module.PAYLOAD_BITS = payload_bits
    return module


def complex_noise(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    scale = 1.0 / math.sqrt(2.0)
    real = torch.randn(shape, device=device, generator=generator) * scale
    imag = torch.randn(shape, device=device, generator=generator) * scale
    return torch.complex(real, imag)


class OrganizerBaselineLink(nn.Module):
    def __init__(self, module: ModuleType):
        super().__init__()
        self.encoder = module.Encoder()
        self.transmitter = module.Transmitter()
        self.receiver = module.Receiver()

    def forward(
        self,
        channel: torch.Tensor,
        bits: torch.Tensor,
        snr_dl: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        feedback_list: list[torch.Tensor] = []
        for user in range(2):
            feedback = self.encoder(channel[:, user], snr_dl[:, user])
            energy = torch.mean(torch.abs(feedback).square(), dim=1, keepdim=True).clamp_min(1e-9)
            feedback = feedback / torch.sqrt(energy)
            snr_ul = snr_dl[:, user] - 10.0
            noise = complex_noise(tuple(feedback.shape), device=feedback.device, generator=generator)
            feedback_list.append(feedback + noise * torch.sqrt(torch.pow(10.0, -snr_ul / 10.0))[:, None])

        signal, control = self.transmitter(
            [bits[:, user] for user in range(2)],
            feedback_list,
            snr_dl.transpose(0, 1),
        )
        energy = torch.mean(torch.sum(torch.abs(signal).square(), dim=1), dim=1, keepdim=True)
        signal = signal / torch.sqrt(energy.clamp_min(1e-9))[:, :, None]
        outputs: list[torch.Tensor] = []
        for user in range(2):
            received = torch.sum(channel[:, user] * signal[:, None], dim=2)
            noise = complex_noise(tuple(received.shape), device=received.device, generator=generator)
            received = received + noise * torch.sqrt(torch.pow(10.0, -snr_dl[:, user] / 10.0))[:, None, None]
            outputs.append(self.receiver(received, channel[:, user], control, snr_dl[:, user]))
        return torch.stack(outputs, dim=1)


def save_model(link: OrganizerBaselineLink, output_dir: Path, model_design: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(link.encoder.state_dict(), output_dir / "encoder.pth")
    torch.save(link.transmitter.state_dict(), output_dir / "transmitter.pth")
    torch.save(link.receiver.state_dict(), output_dir / "receiver.pth")
    shutil.copy2(model_design, output_dir / "modelDesign.py")


def evaluate(
    link: OrganizerBaselineLink,
    data: ChannelMemmap,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    link.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    scores: list[np.ndarray] = []
    loss_sum = 0.0
    active_count = 0
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            channel = torch.from_numpy(data.read(selected)).to(device)
            batch = channel.shape[0]
            bits = torch.randint(0, 2, (batch, 2, 1152), device=device, dtype=torch.float32, generator=generator)
            snr = -20.0 + 40.0 * torch.rand((batch, 2), device=device, generator=generator)
            logits = link(channel, bits, snr, generator=generator)
            targets = bits[..., : logits.shape[-1]]
            loss_sum += float(nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="sum").item())
            active_count += targets.numel()
            batch_scores = [per_sample_score(bits[:, user], logits[:, user]) for user in range(2)]
            scores.append(torch.stack(batch_scores, dim=1).cpu().numpy().reshape(-1))
    summary = summarize_scores(np.concatenate(scores))
    return {
        "loss": loss_sum / active_count,
        "efficiency": summary.efficiency,
        "fairness": summary.fairness,
        "final": summary.final,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    np_rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    module = load_model_design(args.model_design, args.bits_per_re, args.payload_bits)
    link = OrganizerBaselineLink(module).to(device)
    optimizer = torch.optim.Adam(link.parameters(), lr=args.learning_rate)
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.output_dir / "last_checkpoint.pth", map_location=device, weights_only=True)
        link.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])

    data = ChannelMemmap(args.data)
    split = deterministic_split_indices(len(data), seed=1176)
    train_indices = split["train"]
    validation_indices = split["validation"][: args.validation_samples]
    generator = torch.Generator(device=device).manual_seed(args.seed)
    best: dict[str, float] | None = None
    best_step = start_step
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()

    for step in range(start_step + 1, args.steps + 1):
        link.train()
        selected = np_rng.choice(train_indices, args.batch_size, replace=False)
        channel = torch.from_numpy(data.read(selected)).to(device)
        bits = torch.randint(
            0,
            2,
            (args.batch_size, 2, 1152),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        snr = -20.0 + 40.0 * torch.rand((args.batch_size, 2), device=device, generator=generator)
        logits = link(channel, bits, snr, generator=generator)
        targets = bits[..., : logits.shape[-1]]
        loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        should_validate = step % args.validate_every == 0 or step == args.steps
        if should_validate:
            metrics = evaluate(
                link,
                data,
                validation_indices,
                batch_size=args.validation_batch_size,
                device=device,
                seed=args.seed + 10_000,
            )
            record: dict[str, float | int] = {
                "step": step,
                "train_loss": float(loss.item()),
                **metrics,
            }
            history.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if best is None or metrics["final"] > best["final"]:
                best = metrics
                best_step = step
                save_model(link, args.output_dir, args.model_design)
            torch.save(
                {"step": step, "model": link.state_dict(), "optimizer": optimizer.state_dict()},
                args.output_dir / "last_checkpoint.pth",
            )

    assert best is not None
    result = {
        "method": "organizer baseline, random initialization",
        "bits_per_re": args.bits_per_re,
        "payload_bits_B": args.payload_bits,
        "capacity_bits": 144 * args.bits_per_re,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "requested_steps": args.steps,
        "best_step": best_step,
        "best_validation": best,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "output_dir": str(args.output_dir.resolve()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "experiment_config.json").write_text(
        json.dumps(
            {"NUM_BITS_PER_RE": args.bits_per_re, "PAYLOAD_BITS": args.payload_bits},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

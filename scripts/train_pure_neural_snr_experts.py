from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import time
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
from torch import nn

from oppods.data import ChannelMemmap, deterministic_split_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the V191 pure-neural 5 dB SNR expert bank")
    parser.add_argument(
        "--stage",
        choices=("initialize", "pretrain", "asymmetric", "calibrate"),
        required=True,
    )
    parser.add_argument("--expert-index", type=int, choices=range(8))
    parser.add_argument(
        "--train-components",
        nargs="+",
        choices=("encoder", "transmitter", "receiver"),
        default=("encoder", "transmitter", "receiver"),
    )
    parser.add_argument("--steps", type=int, default=750)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--tail-weight", type=float, default=0.0)
    parser.add_argument("--tail-fraction", type=float, default=0.2)
    parser.add_argument("--context-weight", type=float, default=0.25)
    parser.add_argument(
        "--payload-aware",
        action="store_true",
        help="Train and validate the SNR-dependent prefix declared by OUTPUT_PREFIX_POLICY",
    )
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--validation-samples", type=int, default=256)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1191)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data", type=Path, default=Path("ziliao/data_train/H_train.npz"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("ziliao/modelSubmit"))
    parser.add_argument("--model-design", type=Path, default=Path("research/pure_neural_v191/modelDesign.py"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/pure_neural_v191/modelSubmit"))
    args = parser.parse_args()
    expert_stages = {"pretrain", "asymmetric"}
    if args.stage in expert_stages and args.expert_index is None:
        parser.error(f"--stage {args.stage} requires --expert-index")
    if args.stage not in expert_stages and args.expert_index is not None:
        parser.error("--expert-index is only valid for --stage pretrain or asymmetric")
    if args.tail_weight < 0.0:
        parser.error("--tail-weight must be non-negative")
    if not 0.0 < args.tail_fraction <= 1.0:
        parser.error("--tail-fraction must be in (0, 1]")
    if not 0.0 <= args.context_weight <= 1.0:
        parser.error("--context-weight must be in [0, 1]")
    return args


def load_model_design(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("pure_neural_v191_model_design", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import model design from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
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


class PureNeuralLink(nn.Module):
    def __init__(self, module: ModuleType):
        super().__init__()
        self.encoder = module.Encoder()
        self.transmitter = module.Transmitter()
        self.receiver = module.Receiver()

    def initialize_from_baseline(self, baseline_dir: Path) -> None:
        self.encoder.initialize_from_baseline(
            torch.load(baseline_dir / "encoder.pth", map_location="cpu", weights_only=True)
        )
        self.transmitter.initialize_from_baseline(
            torch.load(baseline_dir / "transmitter.pth", map_location="cpu", weights_only=True)
        )
        self.receiver.initialize_from_baseline(
            torch.load(baseline_dir / "receiver.pth", map_location="cpu", weights_only=True)
        )

    def load_submission(self, directory: Path) -> None:
        self.encoder.load_state_dict(torch.load(directory / "encoder.pth", map_location="cpu", weights_only=True))
        self.transmitter.load_state_dict(
            torch.load(directory / "transmitter.pth", map_location="cpu", weights_only=True)
        )
        self.receiver.load_state_dict(torch.load(directory / "receiver.pth", map_location="cpu", weights_only=True))

    def save_submission(self, directory: Path, model_design: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.encoder.state_dict(), directory / "encoder.pth")
        torch.save(self.transmitter.state_dict(), directory / "transmitter.pth")
        torch.save(self.receiver.state_dict(), directory / "receiver.pth")
        destination = directory / "modelDesign.py"
        if model_design.resolve() != destination.resolve():
            shutil.copy2(model_design, destination)

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
            feedback = feedback / torch.sqrt(
                torch.mean(torch.abs(feedback).square(), dim=1, keepdim=True).clamp_min(1e-9)
            )
            snr_ul = snr_dl[:, user] - 10.0
            feedback_noise = complex_noise(tuple(feedback.shape), device=feedback.device, generator=generator)
            feedback_list.append(
                feedback + feedback_noise * torch.sqrt(torch.pow(10.0, -snr_ul / 10.0))[:, None]
            )

        signal, control = self.transmitter(
            [bits[:, user] for user in range(2)],
            feedback_list,
            snr_dl.transpose(0, 1),
        )
        energy = torch.mean(torch.sum(torch.abs(signal).square(), dim=1), dim=1, keepdim=True)
        signal = signal / torch.sqrt(energy.clamp_min(1e-9))[:, :, None]
        output: list[torch.Tensor] = []
        for user in range(2):
            received = torch.sum(channel[:, user] * signal[:, None], dim=2)
            downlink_noise = complex_noise(tuple(received.shape), device=received.device, generator=generator)
            received = received + downlink_noise * torch.sqrt(
                torch.pow(10.0, -snr_dl[:, user] / 10.0)
            )[:, None, None]
            output.append(self.receiver(received, channel[:, user], control, snr_dl[:, user]))
        return torch.stack(output, dim=1)


def expert_parameters(
    link: PureNeuralLink,
    expert_index: int,
    components: Iterable[str] = ("encoder", "transmitter", "receiver"),
) -> Iterable[nn.Parameter]:
    selected = set(components)
    if "encoder" in selected:
        yield from link.encoder.experts[expert_index].parameters()
    if "transmitter" in selected:
        yield from link.transmitter.experts[expert_index].parameters()
    if "receiver" in selected:
        yield from link.receiver.experts[expert_index].parameters()


def score_aligned_bce(
    logits: torch.Tensor,
    bits: torch.Tensor,
    *,
    tail_weight: float,
    tail_fraction: float,
) -> torch.Tensor:
    bits = bits[..., : logits.shape[-1]]
    per_link = nn.functional.binary_cross_entropy_with_logits(logits, bits, reduction="none").mean(dim=-1)
    mean_loss = per_link.mean()
    if tail_weight == 0.0:
        return mean_loss
    flat = per_link.reshape(-1)
    tail_count = max(1, math.ceil(tail_fraction * flat.numel()))
    tail_loss = torch.topk(flat, tail_count).values.mean()
    return (mean_loss + tail_weight * tail_loss) / (1.0 + tail_weight)


def payload_lengths(
    snr: torch.Tensor,
    policy: tuple[tuple[float, float, int], ...],
    max_bits: int,
) -> torch.Tensor:
    lengths = torch.full_like(snr, max_bits, dtype=torch.long)
    for low_db, high_db, prefix_bits in policy:
        if not 0 < prefix_bits <= max_bits:
            raise ValueError(f"invalid payload prefix {prefix_bits}; expected 1..{max_bits}")
        selected = (snr >= low_db) & (snr < high_db)
        lengths = torch.where(selected, torch.full_like(lengths, prefix_bits), lengths)
    return lengths


def payload_aligned_bce(
    logits: torch.Tensor,
    bits: torch.Tensor,
    snr: torch.Tensor,
    *,
    policy: tuple[tuple[float, float, int], ...],
    tail_weight: float,
    tail_fraction: float,
) -> torch.Tensor:
    max_bits = logits.shape[-1]
    targets = bits[..., :max_bits]
    lengths = payload_lengths(snr, policy, max_bits)
    positions = torch.arange(max_bits, device=logits.device)
    active = positions < lengths[..., None]
    element_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    per_link = (element_loss * active).sum(dim=-1) / lengths
    mean_loss = per_link.mean()
    if tail_weight == 0.0:
        return mean_loss
    flat = per_link.reshape(-1)
    tail_count = max(1, math.ceil(tail_fraction * flat.numel()))
    tail_loss = torch.topk(flat, tail_count).values.mean()
    return (mean_loss + tail_weight * tail_loss) / (1.0 + tail_weight)


def asymmetric_target_bce(
    logits: torch.Tensor,
    bits: torch.Tensor,
    *,
    context_weight: float,
    tail_weight: float,
    tail_fraction: float,
) -> torch.Tensor:
    bits = bits[..., : logits.shape[-1]]
    batch_size = logits.shape[0]
    rows = torch.arange(batch_size, device=logits.device)
    target_users = rows % 2
    context_users = 1 - target_users
    target_loss = score_aligned_bce(
        logits[rows, target_users].unsqueeze(1),
        bits[rows, target_users].unsqueeze(1),
        tail_weight=tail_weight,
        tail_fraction=tail_fraction,
    )
    context_loss = nn.functional.binary_cross_entropy_with_logits(
        logits[rows, context_users],
        bits[rows, context_users],
    )
    return (target_loss + context_weight * context_loss) / (1.0 + context_weight)


def sample_snr(
    batch_size: int,
    *,
    stage: str,
    expert_index: int | None,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    if stage == "pretrain":
        assert expert_index is not None
        low_db = -20.0 + 5.0 * expert_index
        high_db = low_db + 5.0
        return low_db + (high_db - low_db) * torch.rand(
            (batch_size, 2),
            device=device,
            generator=generator,
        )
    if stage == "asymmetric":
        assert expert_index is not None
        snr = -20.0 + 40.0 * torch.rand(
            (batch_size, 2),
            device=device,
            generator=generator,
        )
        target_users = torch.arange(batch_size, device=device) % 2
        target_snr = -20.0 + 5.0 * expert_index + 5.0 * torch.rand(
            batch_size,
            device=device,
            generator=generator,
        )
        snr[torch.arange(batch_size, device=device), target_users] = target_snr
        return snr
    return -20.0 + 40.0 * torch.rand(
        (batch_size, 2),
        device=device,
        generator=generator,
    )


def evaluate(
    link: PureNeuralLink,
    data: ChannelMemmap,
    indices: np.ndarray,
    *,
    stage: str,
    expert_index: int | None,
    batch_size: int,
    device: torch.device,
    seed: int,
    payload_policy: tuple[tuple[float, float, int], ...] | None = None,
) -> dict[str, float]:
    link.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    generator = torch.Generator(device=device).manual_seed(seed)
    score_batches: list[torch.Tensor] = []
    loss_sum = 0.0
    bit_count = 0
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selected_indices = indices[start : start + batch_size]
            channel = torch.from_numpy(data.read(selected_indices)).to(device)
            batch = channel.shape[0]
            bits = torch.randint(
                0,
                2,
                (batch, 2, 1152),
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
            snr = sample_snr(
                batch,
                stage=stage,
                expert_index=expert_index,
                device=device,
                generator=generator,
            )
            logits = link(channel, bits, snr, generator=generator)
            targets = bits[..., : logits.shape[-1]]
            loss_sum += float(criterion(logits, targets).item())
            bit_count += targets.numel()
            correct_by_bit = (logits >= 0) == (targets >= 0.5)
            if payload_policy is None:
                correct = correct_by_bit.sum(dim=-1)
                score_batches.append(100.0 * correct / targets.shape[-1])
            else:
                lengths = payload_lengths(snr, payload_policy, targets.shape[-1])
                positions = torch.arange(targets.shape[-1], device=device)
                active = positions < lengths[..., None]
                correct = (correct_by_bit * active).sum(dim=-1)
                score_batches.append(
                    100.0 * (correct + 0.5 * (targets.shape[-1] - lengths)) / targets.shape[-1]
                )
    scores = torch.cat(score_batches).cpu().numpy().reshape(-1)
    efficiency = float(np.mean(scores))
    fairness = float(np.percentile(scores, 10))
    return {
        "loss": loss_sum / bit_count,
        "efficiency": efficiency,
        "fairness": fairness,
        "final": 0.7 * efficiency + 0.3 * fairness,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np_rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    module = load_model_design(args.model_design)
    payload_policy = tuple(module.OUTPUT_PREFIX_POLICY) if args.payload_aware else None
    if args.payload_aware and not getattr(module, "PAYLOAD_INPUT_MASKING", False):
        raise ValueError("--payload-aware requires PAYLOAD_INPUT_MASKING = True in modelDesign.py")
    link = PureNeuralLink(module)
    existing = all(
        (args.output_dir / name).exists()
        for name in ("encoder.pth", "transmitter.pth", "receiver.pth")
    )
    if existing:
        link.load_submission(args.output_dir)
        initialization = "existing expert bank"
    else:
        link.initialize_from_baseline(args.baseline_dir)
        initialization = "organizer baseline replicated into eight experts"
    link.to(device)

    if args.stage == "initialize":
        link.save_submission(args.output_dir, args.model_design)
        result = {
            "stage": args.stage,
            "initialization": initialization,
            "parameters": sum(parameter.numel() for parameter in link.parameters()),
            "output_dir": str(args.output_dir.resolve()),
        }
        (args.output_dir / "latest_training.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    data = ChannelMemmap(args.data)
    split = deterministic_split_indices(len(data), seed=1176)
    validation_indices = split["validation"][: args.validation_samples]
    train_indices = split["train"]
    if args.stage in {"pretrain", "asymmetric"}:
        assert args.expert_index is not None
        parameters = list(expert_parameters(link, args.expert_index, args.train_components))
    else:
        parameters = list(link.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    best = evaluate(
        link,
        data,
        validation_indices,
        stage=args.stage,
        expert_index=args.expert_index,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed + 10_000,
        payload_policy=payload_policy,
    )
    best_step = 0
    checks_without_improvement = 0
    link.save_submission(args.output_dir, args.model_design)
    history: list[dict[str, float | int]] = [{"step": 0, **best}]
    started = time.perf_counter()
    link.train()
    for step in range(1, args.steps + 1):
        batch_indices = np_rng.choice(train_indices, args.batch_size, replace=False)
        channel = torch.from_numpy(data.read(batch_indices)).to(device)
        bits = torch.randint(
            0,
            2,
            (args.batch_size, 2, 1152),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        snr = sample_snr(
            args.batch_size,
            stage=args.stage,
            expert_index=args.expert_index,
            device=device,
            generator=generator,
        )
        logits = link(channel, bits, snr, generator=generator)
        if args.payload_aware:
            loss = payload_aligned_bce(
                logits,
                bits,
                snr,
                policy=payload_policy,
                tail_weight=args.tail_weight,
                tail_fraction=args.tail_fraction,
            )
        elif args.stage == "asymmetric":
            loss = asymmetric_target_bce(
                logits,
                bits,
                context_weight=args.context_weight,
                tail_weight=args.tail_weight,
                tail_fraction=args.tail_fraction,
            )
        else:
            loss = score_aligned_bce(
                logits,
                bits,
                tail_weight=args.tail_weight,
                tail_fraction=args.tail_fraction,
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        optimizer.step()

        if step % args.validate_every == 0 or step == args.steps:
            metrics = evaluate(
                link,
                data,
                validation_indices,
                stage=args.stage,
                expert_index=args.expert_index,
                batch_size=args.batch_size,
                device=device,
                seed=args.seed + 10_000,
                payload_policy=payload_policy,
            )
            record: dict[str, float | int] = {"step": step, **metrics}
            history.append(record)
            print(json.dumps(record, ensure_ascii=False))
            if metrics["final"] > best["final"]:
                best = metrics
                best_step = step
                checks_without_improvement = 0
                link.save_submission(args.output_dir, args.model_design)
            else:
                checks_without_improvement += 1
            if checks_without_improvement >= args.patience:
                break
            link.train()

    result = {
        "stage": args.stage,
        "expert_index": args.expert_index,
        "train_components": args.train_components if args.expert_index is not None else None,
        "snr_interval_db": (
            [-20.0 + 5.0 * args.expert_index, -15.0 + 5.0 * args.expert_index]
            if args.expert_index is not None
            else [-20.0, 20.0]
        ),
        "initialization": initialization,
        "seed": args.seed,
        "tail_weight": args.tail_weight,
        "tail_fraction": args.tail_fraction,
        "context_weight": args.context_weight if args.stage == "asymmetric" else None,
        "payload_aware": args.payload_aware,
        "payload_policy": payload_policy,
        "requested_steps": args.steps,
        "best_step": best_step,
        "best_validation": best,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "output_dir": str(args.output_dir.resolve()),
    }
    if args.stage in {"pretrain", "asymmetric"}:
        report_name = f"{args.stage}_expert_{args.expert_index}.json"
    else:
        report_name = "calibration.json"
    report_path = args.output_dir / report_name
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "latest_training.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

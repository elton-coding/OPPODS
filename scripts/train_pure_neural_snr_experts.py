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
    parser = argparse.ArgumentParser(description="Train the V220 wide residual-MLP link")
    parser.add_argument(
        "--stage",
        choices=("initialize", "pretrain", "asymmetric", "profile", "calibrate"),
        required=True,
    )
    parser.add_argument("--expert-index", type=int, choices=range(1))
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
    parser.add_argument(
        "--loss-kind",
        choices=("bce", "hinge_bce", "wrong_side"),
        default="bce",
    )
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--context-weight", type=float, default=0.25)
    parser.add_argument(
        "--focus-snr-high",
        type=float,
        default=20.0,
        help="Upper bound of the focused low-SNR replay interval",
    )
    parser.add_argument(
        "--focus-prob",
        type=float,
        default=0.0,
        help="Per-user probability of sampling SNR from [-20, focus-snr-high]",
    )
    parser.add_argument(
        "--shared-frontend",
        action="store_true",
        help="Use expert 0 as an identical shared Encoder/Transmitter frontend during training",
    )
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--validation-samples", type=int, default=256)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1191)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data", type=Path, default=Path("ziliao/data_train/H_train.npz"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("ziliao/modelSubmit"))
    parser.add_argument("--model-design", type=Path, default=Path("research/pure_neural_v220/modelDesign.py"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/pure_neural_v220/modelSubmit"))
    args = parser.parse_args()
    if args.stage in {"pretrain", "asymmetric", "profile"} and args.expert_index is None:
        parser.error(f"--stage {args.stage} requires --expert-index")
    if args.stage not in {"pretrain", "asymmetric", "profile"} and args.expert_index is not None:
        parser.error("--expert-index is only valid for expert training")
    if args.tail_weight < 0.0:
        parser.error("--tail-weight must be non-negative")
    if args.margin < 0.0:
        parser.error("--margin must be non-negative")
    if not 0.0 < args.tail_fraction <= 1.0:
        parser.error("--tail-fraction must be in (0, 1]")
    if not 0.0 <= args.context_weight <= 1.0:
        parser.error("--context-weight must be in [0, 1]")
    if not -20.0 < args.focus_snr_high <= 20.0:
        parser.error("--focus-snr-high must be in (-20, 20]")
    if not 0.0 <= args.focus_prob <= 1.0:
        parser.error("--focus-prob must be in [0, 1]")
    if args.shared_frontend and set(args.train_components) != {"receiver"}:
        parser.error("--shared-frontend requires --train-components receiver")
    return args


def load_model_design(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("pure_neural_v220_model_design", path)
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
        def core_state(filename: str) -> dict[str, torch.Tensor]:
            state = torch.load(baseline_dir / filename, map_location="cpu", weights_only=True)
            prefix = "experts.0."
            if state and all(name.startswith(prefix) for name in state):
                return {name[len(prefix):]: value for name, value in state.items()}
            return state

        self.encoder.initialize_from_baseline(core_state("encoder.pth"))
        self.transmitter.initialize_from_baseline(core_state("transmitter.pth"))
        self.receiver.initialize_from_baseline(core_state("receiver.pth"))

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
        shutil.copy2(model_design, directory / "modelDesign.py")

    def forward(
        self,
        channel: torch.Tensor,
        bits: torch.Tensor,
        snr_dl: torch.Tensor,
        *,
        generator: torch.Generator,
        shared_frontend: bool = False,
    ) -> torch.Tensor:
        feedback_list: list[torch.Tensor] = []
        for user in range(2):
            if shared_frontend:
                feedback = self.encoder.experts[0](channel[:, user], snr_dl[:, user])
            else:
                feedback = self.encoder(channel[:, user], snr_dl[:, user])
            feedback = feedback / torch.sqrt(
                torch.mean(torch.abs(feedback).square(), dim=1, keepdim=True).clamp_min(1e-9)
            )
            snr_ul = snr_dl[:, user] - 10.0
            feedback_noise = complex_noise(tuple(feedback.shape), device=feedback.device, generator=generator)
            feedback_list.append(
                feedback + feedback_noise * torch.sqrt(torch.pow(10.0, -snr_ul / 10.0))[:, None]
            )

        transmitter = self.transmitter.experts[0] if shared_frontend else self.transmitter
        signal, control = transmitter(
            [bits[:, user] for user in range(2)], feedback_list, snr_dl.transpose(0, 1)
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
    targets = bits[..., : logits.shape[-1]]
    per_link = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(dim=-1)
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
    targets = bits[..., : logits.shape[-1]]
    rows = torch.arange(logits.shape[0], device=logits.device)
    target_users = rows % 2
    context_users = 1 - target_users
    target_loss = score_aligned_bce(
        logits[rows, target_users].unsqueeze(1),
        targets[rows, target_users].unsqueeze(1),
        tail_weight=tail_weight,
        tail_fraction=tail_fraction,
    )
    context_loss = nn.functional.binary_cross_entropy_with_logits(
        logits[rows, context_users], targets[rows, context_users]
    )
    return (target_loss + context_weight * context_loss) / (1.0 + context_weight)


def score_aligned_loss(
    logits: torch.Tensor,
    bits: torch.Tensor,
    *,
    loss_kind: str,
    margin: float,
    tail_weight: float,
    tail_fraction: float,
) -> torch.Tensor:
    targets = bits[..., : logits.shape[-1]]
    signed_logits = (2.0 * targets - 1.0) * logits
    if loss_kind == "bce":
        element_loss = nn.functional.softplus(-signed_logits)
    elif loss_kind == "hinge_bce":
        bce = nn.functional.softplus(-signed_logits)
        hinge = nn.functional.relu(margin - signed_logits)
        element_loss = 0.5 * bce + 0.5 * hinge
    elif loss_kind == "wrong_side":
        element_loss = nn.functional.relu(-signed_logits)
    else:
        raise ValueError(f"unknown loss kind {loss_kind!r}")
    per_link = element_loss.mean(dim=-1)
    mean_loss = per_link.mean()
    if tail_weight == 0.0:
        return mean_loss
    flat = per_link.reshape(-1)
    tail_count = max(1, math.ceil(tail_fraction * flat.numel()))
    tail_loss = torch.topk(flat, tail_count).values.mean()
    return (mean_loss + tail_weight * tail_loss) / (1.0 + tail_weight)


def sample_snr(
    batch_size: int,
    *,
    stage: str,
    expert_index: int | None,
    device: torch.device,
    generator: torch.Generator,
    focus_snr_high: float = 20.0,
    focus_prob: float = 0.0,
) -> torch.Tensor:
    if stage == "pretrain":
        assert expert_index is not None
        low_db = -20.0 + 2.5 * expert_index
        high_db = low_db + 2.5
        return low_db + (high_db - low_db) * torch.rand(
            (batch_size, 2), device=device, generator=generator
        )
    if stage == "asymmetric":
        assert expert_index is not None
        snr = -20.0 + 40.0 * torch.rand((batch_size, 2), device=device, generator=generator)
        rows = torch.arange(batch_size, device=device)
        target_users = rows % 2
        low_db = -20.0 + 2.5 * expert_index
        snr[rows, target_users] = low_db + 2.5 * torch.rand(
            batch_size, device=device, generator=generator
        )
        return snr
    if stage == "profile":
        assert expert_index is not None
        rows = torch.arange(batch_size, device=device)
        low_db = -20.0 + 2.5 * expert_index
        minimum = low_db + 2.5 * torch.rand(batch_size, device=device, generator=generator)
        partner = minimum + (20.0 - minimum) * torch.rand(
            batch_size, device=device, generator=generator
        )
        snr = torch.empty((batch_size, 2), device=device)
        minimum_user = rows % 2
        snr[rows, minimum_user] = minimum
        snr[rows, 1 - minimum_user] = partner
        return snr
    snr = -20.0 + 40.0 * torch.rand((batch_size, 2), device=device, generator=generator)
    if focus_prob > 0.0:
        focus_mask = torch.rand((batch_size, 2), device=device, generator=generator) < focus_prob
        focused = -20.0 + (focus_snr_high + 20.0) * torch.rand(
            (batch_size, 2), device=device, generator=generator
        )
        snr = torch.where(focus_mask, focused, snr)
    return snr


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
    shared_frontend: bool = False,
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
            logits = link(channel, bits, snr, generator=generator, shared_frontend=shared_frontend)
            targets = bits[..., : logits.shape[-1]]
            loss_sum += float(criterion(logits, targets).item())
            bit_count += targets.numel()
            correct = ((logits >= 0) == (targets >= 0.5)).sum(dim=-1)
            missing = bits.shape[-1] - targets.shape[-1]
            score_batches.append(100.0 * (correct + 0.5 * missing) / bits.shape[-1])
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
        initialization = f"baseline replicated into {module.NUM_EXPERTS} experts"
    link.to(device)

    if args.stage == "initialize":
        link.save_submission(args.output_dir, args.model_design)
        result = {
            "stage": args.stage,
            "initialization": initialization,
            "parameters": sum(parameter.numel() for parameter in link.parameters()),
            "output_dir": str(args.output_dir.resolve()),
        }
        (args.output_dir.parent / "latest_training.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    data = ChannelMemmap(args.data)
    split = deterministic_split_indices(len(data), seed=1176)
    validation_indices = split["validation"][: args.validation_samples]
    train_indices = split["train"]
    if args.stage in {"pretrain", "asymmetric", "profile"}:
        assert args.expert_index is not None
        parameters = list(expert_parameters(link, args.expert_index, args.train_components))
    else:
        selected = set(args.train_components)
        parameters = []
        if "encoder" in selected:
            parameters.extend(link.encoder.parameters())
        if "transmitter" in selected:
            parameters.extend(link.transmitter.parameters())
        if "receiver" in selected:
            parameters.extend(link.receiver.parameters())
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
        shared_frontend=args.shared_frontend,
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
            focus_snr_high=args.focus_snr_high,
            focus_prob=args.focus_prob,
        )
        logits = link(
            channel,
            bits,
            snr,
            generator=generator,
            shared_frontend=args.shared_frontend,
        )
        if args.stage == "asymmetric":
            loss = asymmetric_target_bce(
                logits,
                bits,
                context_weight=args.context_weight,
                tail_weight=args.tail_weight,
                tail_fraction=args.tail_fraction,
            )
        else:
            loss = score_aligned_loss(
                logits,
                bits,
                loss_kind=args.loss_kind,
                margin=args.margin,
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
                shared_frontend=args.shared_frontend,
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
        "train_components": list(args.train_components) if args.expert_index is not None else None,
        "snr_interval_db": (
            [-20.0 + 2.5 * args.expert_index, -17.5 + 2.5 * args.expert_index]
            if args.expert_index is not None
            else [-20.0, 20.0]
        ),
        "initialization": initialization,
        "seed": args.seed,
        "tail_weight": args.tail_weight,
        "tail_fraction": args.tail_fraction,
        "loss_kind": args.loss_kind,
        "margin": args.margin,
        "context_weight": args.context_weight if args.stage == "asymmetric" else None,
        "focus_snr_high": args.focus_snr_high,
        "focus_prob": args.focus_prob,
        "shared_frontend": args.shared_frontend,
        "requested_steps": args.steps,
        "best_step": best_step,
        "best_validation": best,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "output_dir": str(args.output_dir.resolve()),
    }
    report_path = args.output_dir.parent / (
        f"{args.stage}_expert_{args.expert_index}.json"
        if args.stage in {"pretrain", "asymmetric", "profile"}
        else "calibration.json"
    )
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir.parent / "latest_training.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

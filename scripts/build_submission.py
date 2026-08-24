from __future__ import annotations

import argparse
import importlib.util
import json
import zipfile
from pathlib import Path
from types import ModuleType

import torch

ARCHIVE_MEMBERS = (
    "submit_pt/modelDesign.py",
    "submit_pt/modelSubmit/encoder.pth",
    "submit_pt/modelSubmit/transmitter.pth",
    "submit_pt/modelSubmit/receiver.pth",
)


def load_model_design() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "modelSubmit" / "modelDesign.py"
    spec = importlib.util.spec_from_file_location("submission_model_design", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the DataFountain submission weights and archive")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/sparse_denoiser_task.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("modelSubmit"))
    parser.add_argument("--archive", type=Path, default=Path("artifacts/FATE_MIMO_submission.zip"))
    parser.add_argument(
        "--package-only",
        action="store_true",
        help="Package the already versioned modelSubmit files without rebuilding weights from a checkpoint.",
    )
    return parser.parse_args()


def package_submission(output_dir: Path, archive_path: Path) -> dict[str, int]:
    sources = {
        "submit_pt/modelDesign.py": output_dir / "modelDesign.py",
        "submit_pt/modelSubmit/encoder.pth": output_dir / "encoder.pth",
        "submit_pt/modelSubmit/transmitter.pth": output_dir / "transmitter.pth",
        "submit_pt/modelSubmit/receiver.pth": output_dir / "receiver.pth",
    }
    missing = [str(source) for source in sources.values() if not source.is_file()]
    if missing:
        raise FileNotFoundError(f"submission files are missing: {missing}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for archive_name in ARCHIVE_MEMBERS:
            archive.write(sources[archive_name], arcname=archive_name)
    return {archive_name: sources[archive_name].stat().st_size for archive_name in ARCHIVE_MEMBERS}


def main() -> None:
    args = parse_args()
    checkpoint = None
    if not args.package_only:
        model_design = load_model_design()
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        encoder = model_design.Encoder()
        transmitter = model_design.Transmitter()
        receiver = model_design.Receiver()
        experts = [transmitter.decoder, *getattr(transmitter, "expert_decoders", [])]
        expert_states = checkpoint.get("experts")
        if expert_states is None:
            expert_states = [checkpoint["denoiser"]] * len(experts)
        if len(expert_states) != len(experts):
            raise ValueError(f"checkpoint has {len(expert_states)} experts, model expects {len(experts)}")
        for expert, state in zip(experts, expert_states, strict=True):
            expert.load_state_dict(state)
        if "receiver" in checkpoint:
            receiver.load_state_dict(checkpoint["receiver"])
        args.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(encoder.state_dict(), args.output_dir / "encoder.pth")
        torch.save(transmitter.state_dict(), args.output_dir / "transmitter.pth")
        torch.save(receiver.state_dict(), args.output_dir / "receiver.pth")

    members = package_submission(args.output_dir, args.archive)
    result = {
        "mode": "package_only" if args.package_only else "checkpoint_build",
        "archive": str(args.archive.resolve()),
        "archive_bytes": args.archive.stat().st_size,
        "members": members,
    }
    if checkpoint is not None:
        result.update(
            {
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_step": checkpoint["step"],
                "checkpoint_validation_loss": checkpoint["best_validation"],
            }
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

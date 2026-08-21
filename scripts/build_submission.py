from __future__ import annotations

import argparse
import importlib.util
import json
import zipfile
from pathlib import Path
from types import ModuleType

import torch


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_design = load_model_design()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    encoder = model_design.Encoder()
    transmitter = model_design.Transmitter()
    receiver = model_design.Receiver()
    transmitter.decoder.load_state_dict(checkpoint["denoiser"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), args.output_dir / "encoder.pth")
    torch.save(transmitter.state_dict(), args.output_dir / "transmitter.pth")
    torch.save(receiver.state_dict(), args.output_dir / "receiver.pth")

    args.archive.parent.mkdir(parents=True, exist_ok=True)
    members = {
        "submit_pt/modelDesign.py": args.output_dir / "modelDesign.py",
        "submit_pt/modelSubmit/encoder.pth": args.output_dir / "encoder.pth",
        "submit_pt/modelSubmit/transmitter.pth": args.output_dir / "transmitter.pth",
        "submit_pt/modelSubmit/receiver.pth": args.output_dir / "receiver.pth",
    }
    with zipfile.ZipFile(args.archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for archive_name, source in members.items():
            archive.write(source, arcname=archive_name)
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": checkpoint["step"],
        "checkpoint_validation_loss": checkpoint["best_validation"],
        "archive": str(args.archive.resolve()),
        "archive_bytes": args.archive.stat().st_size,
        "members": {archive_name: source.stat().st_size for archive_name, source in members.items()},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

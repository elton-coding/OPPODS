from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

SNR_INTERVALS_DB = (
    (-20.0, -15.0),
    (-15.0, -10.0),
    (-10.0, -5.0),
    (-5.0, 0.0),
    (0.0, 5.0),
    (5.0, 10.0),
    (10.0, 15.0),
    (15.0, 20.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bundle eight independently trained 5 dB SNR experts")
    parser.add_argument("--checkpoints", type=Path, nargs=8, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoints = [torch.load(path, map_location="cpu", weights_only=False) for path in args.checkpoints]
    reference = checkpoints[0]["denoiser"]
    for index, checkpoint in enumerate(checkpoints[1:], start=1):
        if checkpoint["denoiser"].keys() != reference.keys():
            raise ValueError(f"expert {index} has incompatible state dictionary keys")
        for name, value in checkpoint["denoiser"].items():
            if value.shape != reference[name].shape or value.dtype != reference[name].dtype:
                raise ValueError(f"expert {index} has incompatible tensor {name}")
    output = {
        "step": max(int(checkpoint.get("step", -1)) for checkpoint in checkpoints),
        "best_validation": [checkpoint.get("best_validation") for checkpoint in checkpoints],
        "denoiser": reference,
        "experts": [checkpoint["denoiser"] for checkpoint in checkpoints],
        "config": {
            "snr_intervals_db": SNR_INTERVALS_DB,
            "source_checkpoints": [str(path) for path in args.checkpoints],
            "source_configs": [checkpoint.get("config") for checkpoint in checkpoints],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "experts": len(checkpoints),
                "snr_intervals_db": SNR_INTERVALS_DB,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

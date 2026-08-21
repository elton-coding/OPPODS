from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linearly average compatible denoiser checkpoints")
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument("--secondary-weight", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.secondary_weight <= 1.0:
        raise ValueError("secondary weight must be in [0, 1]")

    primary = torch.load(args.primary, map_location="cpu", weights_only=False)
    secondary = torch.load(args.secondary, map_location="cpu", weights_only=False)
    primary_state = primary["denoiser"]
    secondary_state = secondary["denoiser"]
    if primary_state.keys() != secondary_state.keys():
        raise ValueError("checkpoint parameter sets do not match")

    weight = args.secondary_weight
    averaged = {
        name: (1.0 - weight) * primary_state[name] + weight * secondary_state[name]
        for name in primary_state
    }
    result = dict(primary)
    result["denoiser"] = averaged
    result["config"] = {
        **primary.get("config", {}),
        "checkpoint_average": {
            "primary": str(args.primary),
            "secondary": str(args.secondary),
            "secondary_weight": weight,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interpolate two sparse-denoiser checkpoints")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    base = torch.load(args.base, map_location="cpu", weights_only=False)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=False)
    base_state = base["denoiser"]
    candidate_state = candidate["denoiser"]
    if base_state.keys() != candidate_state.keys():
        raise ValueError("checkpoint state dictionaries do not match")
    interpolated = {
        name: base_tensor.lerp(candidate_state[name], args.alpha)
        if torch.is_floating_point(base_tensor)
        else base_tensor.clone()
        for name, base_tensor in base_state.items()
    }
    output = {
        "step": candidate.get("step", -1),
        "best_validation": candidate.get("best_validation", float("nan")),
        "denoiser": interpolated,
        "config": {
            "interpolation": {
                "base": str(args.base),
                "candidate": str(args.candidate),
                "alpha": args.alpha,
            }
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scale a denoiser's effective residual branch")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--factor", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.factor <= 0.0:
        raise ValueError("factor must be positive")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = payload["denoiser"]
    raw = state["residual_scale"].to(torch.float64)
    effective = torch.tanh(raw)
    target = (args.factor * effective).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    state["residual_scale"] = torch.atanh(target).to(state["residual_scale"].dtype)
    config = dict(payload.get("config", {}))
    config["residual_scale_factor"] = args.factor
    payload["config"] = config
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(
        f"{args.output.resolve()} effective={math.tanh(float(raw)):.8f} "
        f"target={float(target):.8f}"
    )


if __name__ == "__main__":
    main()

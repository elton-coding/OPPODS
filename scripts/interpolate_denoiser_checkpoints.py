from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interpolate two sparse-denoiser checkpoints")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True, help="Candidate weight in [0, 1]")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def interpolate_state_dicts(
    base: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor], alpha: float
) -> dict[str, torch.Tensor]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if base.keys() != candidate.keys():
        raise ValueError("checkpoint state dictionaries have different keys")
    output: dict[str, torch.Tensor] = {}
    for name, base_value in base.items():
        candidate_value = candidate[name]
        if base_value.shape != candidate_value.shape or base_value.dtype != candidate_value.dtype:
            raise ValueError(f"incompatible tensor {name}")
        if base_value.is_floating_point() or base_value.is_complex():
            output[name] = torch.lerp(base_value, candidate_value, alpha)
        else:
            if not torch.equal(base_value, candidate_value):
                raise ValueError(f"non-floating tensor differs: {name}")
            output[name] = base_value.clone()
    return output


def main() -> None:
    args = parse_args()
    base = torch.load(args.base, map_location="cpu", weights_only=False)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=False)
    denoiser = interpolate_state_dicts(base["denoiser"], candidate["denoiser"], args.alpha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "step": candidate.get("step"),
        "best_validation": candidate.get("best_validation"),
        "denoiser": denoiser,
        "config": {
            "interpolation": {
                "base": str(args.base),
                "candidate": str(args.candidate),
                "candidate_alpha": args.alpha,
            },
            "candidate_config": candidate.get("config"),
        },
    }
    torch.save(metadata, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "candidate_alpha": args.alpha,
                "tensors": len(denoiser),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

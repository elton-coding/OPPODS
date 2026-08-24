from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average compatible Receiver-expert checkpoints")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def average_state_dicts(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("at least one state dictionary is required")
    keys = states[0].keys()
    if any(state.keys() != keys for state in states[1:]):
        raise ValueError("state dictionaries have different keys")
    averaged: dict[str, torch.Tensor] = {}
    for name in keys:
        values = [state[name] for state in states]
        reference = values[0]
        if any(value.shape != reference.shape or value.dtype != reference.dtype for value in values[1:]):
            raise ValueError(f"incompatible tensor {name}")
        if reference.is_floating_point() or reference.is_complex():
            total = torch.zeros_like(reference)
            for value in values:
                total = total + value
            averaged[name] = total / len(values)
        else:
            if any(not torch.equal(value, reference) for value in values[1:]):
                raise ValueError(f"non-floating tensor differs: {name}")
            averaged[name] = reference.clone()
    return averaged


def main() -> None:
    args = parse_args()
    checkpoints = [torch.load(path, map_location="cpu", weights_only=False) for path in args.inputs]
    receiver = average_state_dicts([checkpoint["receiver"] for checkpoint in checkpoints])
    output = dict(checkpoints[0])
    output["receiver"] = receiver
    output["step"] = max(int(checkpoint.get("step", -1)) for checkpoint in checkpoints)
    output["best_validation"] = [checkpoint.get("best_validation") for checkpoint in checkpoints]
    output["config"] = {
        "average_receiver_checkpoints": [str(path) for path in args.inputs],
        "source_configs": [checkpoint.get("config") for checkpoint in checkpoints],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(
        json.dumps(
            {"output": str(args.output.resolve()), "inputs": len(checkpoints), "receiver_tensors": len(receiver)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

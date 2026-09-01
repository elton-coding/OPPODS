from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert an 8-bit/RE pure-neural bank to a smaller payload")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--bits-per-re", type=int, choices=range(1, 9), required=True)
    parser.add_argument("--model-design", type=Path, default=Path("research/pure_neural_v191/modelDesign.py"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bits_per_re = args.bits_per_re
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.base / "encoder.pth", args.output / "encoder.pth")

    transmitter = torch.load(args.base / "transmitter.pth", map_location="cpu", weights_only=True)
    transmitter_slices = 0
    for name, tensor in transmitter.items():
        if name.endswith(("._mod.0.0.weight", "._mod.1.0.weight")):
            if tensor.shape[1] != 8:
                raise ValueError(f"expected an 8-bit modulator input in {name}, got {tuple(tensor.shape)}")
            transmitter[name] = tensor[:, :bits_per_re].clone()
            transmitter_slices += 1
    if transmitter_slices != 16:
        raise ValueError(f"expected 16 modulator input tensors, found {transmitter_slices}")
    torch.save(transmitter, args.output / "transmitter.pth")

    receiver = torch.load(args.base / "receiver.pth", map_location="cpu", weights_only=True)
    receiver_slices = 0
    for name, tensor in receiver.items():
        if name.endswith("._fc_out.weight"):
            if tensor.shape[0] != 8:
                raise ValueError(f"expected an 8-bit receiver head in {name}, got {tuple(tensor.shape)}")
            receiver[name] = tensor[:bits_per_re].clone()
            receiver_slices += 1
        elif name.endswith("._fc_out.bias"):
            if tensor.shape[0] != 8:
                raise ValueError(f"expected an 8-bit receiver bias in {name}, got {tuple(tensor.shape)}")
            receiver[name] = tensor[:bits_per_re].clone()
            receiver_slices += 1
    if receiver_slices != 16:
        raise ValueError(f"expected 16 receiver output tensors, found {receiver_slices}")
    torch.save(receiver, args.output / "receiver.pth")

    source = args.model_design.read_text(encoding="utf-8")
    source, replacements = re.subn(
        r"^NUM_BITS_PER_SYMBOL = \d+$",
        f"NUM_BITS_PER_SYMBOL = {bits_per_re}",
        source,
        count=1,
        flags=re.MULTILINE,
    )
    if replacements != 1:
        raise ValueError("could not replace NUM_BITS_PER_SYMBOL in modelDesign.py")
    (args.output / "modelDesign.py").write_text(source, encoding="utf-8")

    result = {
        "base": str(args.base.resolve()),
        "bits_per_re": bits_per_re,
        "full_payload_bits": 144 * bits_per_re,
        "transmitter_slices": transmitter_slices,
        "receiver_slices": receiver_slices,
        "output": str(args.output.resolve()),
    }
    report = args.output.parent / f"{args.output.name}_conversion.json"
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

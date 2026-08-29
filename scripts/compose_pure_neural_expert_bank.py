from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

WEIGHT_FILES = ("encoder.pth", "transmitter.pth", "receiver.pth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replace selected V191 experts with weights from another bank")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--experts", type=int, nargs="+", choices=range(8), required=True)
    parser.add_argument(
        "--components",
        nargs="+",
        choices=("encoder", "transmitter", "receiver"),
        default=("encoder", "transmitter", "receiver"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    replaced: dict[str, int] = {}
    for filename in WEIGHT_FILES:
        base = torch.load(args.base / filename, map_location="cpu", weights_only=True)
        component = Path(filename).stem
        if component not in args.components:
            torch.save(base, args.output / filename)
            replaced[filename] = 0
            continue
        donor = torch.load(args.donor / filename, map_location="cpu", weights_only=True)
        if base.keys() != donor.keys():
            raise ValueError(f"incompatible state dictionaries for {filename}")
        count = 0
        for expert_index in args.experts:
            prefix = f"experts.{expert_index}."
            names = [name for name in base if name.startswith(prefix)]
            if not names:
                raise ValueError(f"{filename} has no tensors for expert {expert_index}")
            for name in names:
                if base[name].shape != donor[name].shape or base[name].dtype != donor[name].dtype:
                    raise ValueError(f"incompatible tensor {name} in {filename}")
                base[name] = donor[name].clone()
                count += 1
        torch.save(base, args.output / filename)
        replaced[filename] = count
    shutil.copy2(args.base / "modelDesign.py", args.output / "modelDesign.py")
    result = {
        "base": str(args.base.resolve()),
        "donor": str(args.donor.resolve()),
        "experts": sorted(set(args.experts)),
        "components": sorted(set(args.components)),
        "output": str(args.output.resolve()),
        "replaced_tensors": replaced,
    }
    (args.output.parent / f"{args.output.name}_composition.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

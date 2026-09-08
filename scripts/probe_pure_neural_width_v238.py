"""Check V238's real V227 warm start on synthetic CPU inputs, not a score test."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch

from run_pure_neural_lr_v237 import ROOT, fingerprints
from train_pure_neural_snr_experts import PureNeuralLink


def load_design(version: int):
    path = ROOT / f"research/pure_neural_v{version}/modelDesign.py"
    spec = importlib.util.spec_from_file_location(f"v{version}_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("probe report already exists")
    torch.set_num_threads(2)
    torch.manual_seed(238)
    parent_dir = ROOT / "artifacts/pure_neural_v227/joint_low"
    paths = [parent_dir / name for name in ("encoder.pth", "transmitter.pth", "receiver.pth")]
    paths += [ROOT / "research/pure_neural_v238/modelDesign.py"]
    before = fingerprints(paths)
    parent = PureNeuralLink(load_design(227)).eval()
    parent.initialize_from_expert_bank(parent_dir, [0, 1])
    candidate = PureNeuralLink(load_design(238)).eval()
    candidate.initialize_from_expert_bank(parent_dir, [0, 1])
    h = torch.randn(3, 2, 2, 16, 144, dtype=torch.complex64)
    bits = torch.randint(0, 2, (3, 2, 1152)).float()
    snr = torch.tensor([[-18., 12.], [-8., 3.], [15., 19.]])
    with torch.inference_mode():
        original = parent(h, bits, snr, generator=torch.Generator().manual_seed(15240))
        widened = candidate(h, bits, snr, generator=torch.Generator().manual_seed(15240))
    torch.testing.assert_close(widened, original, atol=2e-4, rtol=2e-5)
    assert fingerprints(paths) == before
    result = {
        "device": "cpu", "synthetic_seed": 238, "link_noise_seed": 15240,
        "snr": snr.tolist(), "logits": original.numel(),
        "maximum_absolute_logit_difference": (widened - original).abs().max().item(),
        "mean_absolute_logit_difference": (widened - original).abs().mean().item(),
        "hard_decision_disagreements": int(((widened >= 0) != (original >= 0)).sum()),
        "parameters": sum(p.numel() for p in candidate.parameters()),
        "input_sha256": before,
        "caveat": "Synthetic input CPU warm-start parity only; not validation, a trained result, or competition score.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

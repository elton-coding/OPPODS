"""Read-only validation diagnostic: soft-score bias versus hard bit decisions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from oppods.data import ChannelMemmap, deterministic_split_indices
from train_pure_neural_snr_experts import PureNeuralLink, load_model_design, sample_snr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.08, torch.cuda.current_device())
    device = torch.device("cuda")
    link = PureNeuralLink(load_model_design(args.submission / "modelDesign.py"))
    link.load_submission(args.submission)
    link.to(device).eval()
    data = ChannelMemmap(Path("ziliao/data_train/H_train.npz"))
    indices = deterministic_split_indices(len(data), seed=1176)["validation"][:args.samples]
    generator = torch.Generator(device=device).manual_seed(25240)
    margins, hard, snrs = [], [], []
    with torch.inference_mode():
        for start in range(0, len(indices), args.batch_size):
            channel = torch.from_numpy(data.read(indices[start:start + args.batch_size])).to(device)
            batch = len(channel)
            bits = torch.randint(0, 2, (batch, 2, 1152), device=device,
                                 dtype=torch.float32, generator=generator)
            snr = sample_snr(batch, stage="calibrate", expert_index=None,
                             device=device, generator=generator)
            logits = link(channel, bits, snr, generator=generator)
            margins.append(((2 * bits - 1) * logits).cpu())
            hard.append(((logits >= 0) == (bits >= .5)).float().cpu())
            snrs.append(snr.cpu())
    signed = torch.cat(margins).reshape(-1, 1152)
    hard_correct = torch.cat(hard).reshape(-1, 1152)
    own_snr = torch.cat(snrs).flatten()
    hard_scores = hard_correct.mean(-1).numpy() * 100

    def scores(values):
        mean, p10 = float(values.mean()), float(np.percentile(values, 10))
        return {"efficiency": mean, "p10": p10, "final": .7 * mean + .3 * p10}

    comparisons = []
    for temperature in (.5, .25, .1, .05):
        soft = torch.sigmoid(signed / temperature)
        soft_scores = soft.mean(-1).numpy() * 100
        comparisons.append({"temperature": temperature, "global_soft_score": scores(soft_scores),
                            "per_ue_score_mae": float(np.mean(np.abs(soft_scores - hard_scores))),
                            "per_ue_pearson": float(np.corrcoef(soft_scores, hard_scores)[0, 1]),
                            "mean_sigmoid_derivative": float((soft * (1 - soft) / temperature).mean())})
    result = {"submission": str(args.submission), "partition": "split1176 validation only",
              "seed": 25240, "samples": len(indices), "batch_size": args.batch_size,
              "hard_score": scores(hard_scores), "temperature_diagnostics": comparisons,
              "confident_wrong_fraction_by_own_snr": {
                  f"[{lo},{lo+5})": float((signed[(own_snr >= lo) & (own_snr < lo+5)] < -1).float().mean())
                  for lo in range(-20, 20, 5)},
              "caveat": "Diagnostic only, not official batch1 evaluation. Temperature changes do not change this frozen model's hard scores; quantiles here are global, not minibatch rank-smoothed training loss."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

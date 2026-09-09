"""Real V227 CPU gradient equivalence; no held-out channel data or performance score."""
from __future__ import annotations

import json

import torch
import train_pure_neural_global_batch_v246 as entry
from run_pure_neural_lr_v237 import ROOT, fingerprints


def main():
    path = ROOT / "benchmarks/v246_parent_cpu_gradient_probe.json"
    if path.exists():
        raise FileExistsError("probe evidence already exists")
    torch.set_num_threads(2)
    torch.manual_seed(246)
    parent = ROOT / "artifacts/pure_neural_v227/joint_low"
    design = ROOT / "research/pure_neural_v227/modelDesign.py"
    link = entry.trainer.PureNeuralLink(entry.trainer.load_model_design(design))
    link.initialize_from_expert_bank(parent, [0, 1])
    parameters = entry.trainer.select_trainable_parameters(link, ["transmitter", "receiver"])
    link.train()
    channel = torch.randn(4, 2, 2, 16, 144, dtype=torch.complex64)
    bits = torch.randint(0, 2, (4, 2, 1152)).float()
    snr = torch.tensor([[-18., 12.], [0., 8.], [-12., -15.], [18., -4.]])
    reference = None
    max_gradient_delta = 0.
    max_logit_delta = 0.
    for recompute in (False, True):
        link.zero_grad(set_to_none=True)
        generator = torch.Generator().manual_seed(246)
        logits = entry.chunked_forward(link, channel, bits, snr, generator=generator,
                                       microbatch=2, recompute=recompute)
        state = generator.get_state().clone()
        loss = entry.trainer.score_aligned_loss(
            logits, bits, loss_kind="soft_score", margin=.5, tail_weight=0., tail_fraction=.1,
            score_temperature=.5, quantile_bandwidth=.025, score_fairness_weight=.3, score_bce_weight=.05)
        loss.backward()
        if not torch.equal(state, generator.get_state()):
            raise RuntimeError("backward changed the future explicit RNG stream")
        gradients = [p.grad.detach().clone() if p.grad is not None else None for p in parameters]
        if reference is None:
            reference = (logits.detach().clone(), gradients)
        else:
            max_logit_delta = float((logits.detach() - reference[0]).abs().max())
            torch.testing.assert_close(logits, reference[0], atol=0, rtol=0)
            for actual, expected in zip(gradients, reference[1], strict=True):
                if actual is None or expected is None:
                    if actual is not expected:
                        raise RuntimeError("gradient participation changed")
                    continue
                if not torch.isfinite(actual).all():
                    raise RuntimeError("nonfinite gradient")
                max_gradient_delta = max(max_gradient_delta, float((actual - expected).abs().max()))
                torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
    paths = [design, ROOT / "scripts/train_pure_neural_global_batch_v246.py",
             ROOT / "scripts/train_pure_neural_snr_experts.py"]
    paths += [parent / name for name in ("encoder.pth", "transmitter.pth", "receiver.pth")]
    report = {"purpose": "gradient/runtime equivalence only; synthetic channels, no score evaluation",
              "global_batch": 4, "microbatch": 2, "trainable_parameters": sum(p.numel() for p in parameters),
              "max_logit_delta": max_logit_delta, "max_gradient_delta": max_gradient_delta,
              "backward_preserves_generator": True, "input_sha256": fingerprints(paths)}
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

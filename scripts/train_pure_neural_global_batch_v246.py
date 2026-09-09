"""Global-batch score loss with RNG-exact checkpointed microbatch forwards.

Only activations are recomputed; the score loss sees all UE logits at once.
Validation keeps the original batch100 random-input stream.
"""
from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager, nullcontext
from functools import partial

import torch
import train_pure_neural_snr_experts as trainer
from torch.utils.checkpoint import checkpoint

BASE_FORWARD = trainer.PureNeuralLink.forward
BASE_PARSE = trainer.parse_args
BASE_EVALUATE = trainer.evaluate


@contextmanager
def replay_generator(generator: torch.Generator, state: torch.Tensor):
    current = generator.get_state()
    generator.set_state(state)
    try:
        yield
    finally:
        generator.set_state(current)


def contexts(generator, state):
    return nullcontext(), replay_generator(generator, state)


def chunked_forward(link, channel, bits, snr, *, generator, shared_frontend=False,
                    microbatch=100, recompute=True):
    if microbatch < 1:
        raise ValueError("microbatch must be positive")
    outputs = []
    for start in range(0, len(channel), microbatch):
        end = start + microbatch
        args = (link, channel[start:end], bits[start:end], snr[start:end])
        options = {"generator": generator, "shared_frontend": shared_frontend}
        if recompute and torch.is_grad_enabled():
            # The explicit channel-noise generator is NOT covered by checkpoint's
            # default RNG preservation. Bind each chunk's own state immediately.
            outputs.append(checkpoint(BASE_FORWARD, *args, **options, use_reentrant=False,
                                      context_fn=partial(contexts, generator, generator.get_state())))
        else:
            outputs.append(BASE_FORWARD(*args, **options))
    return torch.cat(outputs, dim=0)


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--microbatch-size", type=int, default=100)
    extra, remaining = parser.parse_known_args()
    original = sys.argv
    try:
        sys.argv = [original[0], *remaining]
        args = BASE_PARSE()
    finally:
        sys.argv = original
    if args.stage != "calibrate" or args.loss_kind != "soft_score" or extra.microbatch_size < 1:
        raise ValueError("V246 requires calibrate, soft_score and positive microbatch size")
    args.microbatch_size = extra.microbatch_size
    return args


def evaluate(*args, **kwargs):
    kwargs["batch_size"] = 100
    return BASE_EVALUATE(*args, **kwargs)


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError("V246 output exists; refusing implicit resume")
    original = (trainer.parse_args, trainer.PureNeuralLink.forward, trainer.evaluate)

    def forward(link, channel, bits, snr, *, generator, shared_frontend=False):
        if not link.training or not torch.is_grad_enabled():
            return BASE_FORWARD(link, channel, bits, snr, generator=generator, shared_frontend=shared_frontend)
        return chunked_forward(link, channel, bits, snr, generator=generator,
                               shared_frontend=shared_frontend, microbatch=args.microbatch_size)

    try:
        trainer.parse_args = lambda: args
        trainer.PureNeuralLink.forward = forward
        trainer.evaluate = evaluate
        trainer.main()
    finally:
        trainer.parse_args, trainer.PureNeuralLink.forward, trainer.evaluate = original
    path = args.output_dir / "training_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report.update({"microbatch_size": args.microbatch_size, "validation_batch_size": 100,
                   "global_loss_ue_count": 2 * args.batch_size,
                   "training_channel_draws": report["history"][-1]["step"] * args.batch_size,
                   "activation_checkpointing": "nonreentrant; explicit per-chunk generator replay",
                   "random_stream_caveat": "batch grouping and microbatch channel-noise draws differ from batch100 control"})
    for destination in (path, args.output_dir.parent / "calibration.json",
                        args.output_dir.parent / "latest_training.json"):
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

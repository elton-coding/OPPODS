"""V255: unchanged RMS/Adam training, constant36k then cosine36k to one tenth LR."""
from __future__ import annotations

import hashlib
import json
import math
import struct

import torch
import train_pure_neural_rms_v239 as rms

HOLD_STEPS = 36000
TOTAL_STEPS = 72000
INITIAL_LR = 1e-5
FINAL_RATIO = .1


def learning_rate(update):
    if not isinstance(update, int) or not 1 <= update <= TOTAL_STEPS:
        raise ValueError("schedule uses successful optimizer updates numbered1..72000")
    if update <= HOLD_STEPS:
        return INITIAL_LR
    fraction = (update - HOLD_STEPS) / (TOTAL_STEPS - HOLD_STEPS)
    return INITIAL_LR * (FINAL_RATIO + (1. - FINAL_RATIO) * .5 * (1. + math.cos(math.pi * fraction)))


def schedule_definition():
    return {"kind": "constant_then_cosine", "hold_updates": HOLD_STEPS, "total_updates": TOTAL_STEPS,
            "initial_learning_rate": INITIAL_LR, "minimum_ratio": FINAL_RATIO,
            "applied_before_optimizer_step": True, "optimizer_state_restart": False,
            "probe_uses_full72000_schedule_not_two_step_decay": True}


class RateRecorder:
    """Hooks change only the LR of the existing Adam; no new optimizer or RNG calls."""

    def __init__(self, optimizer):
        if len(optimizer.param_groups) != 1 or optimizer.param_groups[0]["lr"] != INITIAL_LR:
            raise ValueError("V255 requires one unchanged Adam group at initial1e-5")
        self.completed = 0
        self.pending = None
        self.trace = []
        self.digest = hashlib.sha256()
        self.last_rate = None
        self.handles = (optimizer.register_step_pre_hook(self.before), optimizer.register_step_post_hook(self.after))

    def before(self, optimizer, args, kwargs):
        if self.pending is not None:
            raise RuntimeError("previous optimizer update did not complete")
        rate = learning_rate(self.completed + 1)
        optimizer.param_groups[0]["lr"] = rate
        self.pending = rate

    def after(self, optimizer, args, kwargs):
        if self.pending is None or optimizer.param_groups[0]["lr"] != self.pending:
            raise RuntimeError("optimizer rate changed outside registered schedule")
        self.completed += 1
        self.last_rate = self.pending
        self.digest.update(struct.pack("<d", self.pending))
        if self.completed in (1, 2, HOLD_STEPS + 1) or self.completed % 1000 == 0:
            self.trace.append({"update": self.completed, "learning_rate": self.pending})
        self.pending = None

    def evidence(self):
        if self.pending is not None:
            raise RuntimeError("incomplete optimizer update cannot be reported as training")
        return {**schedule_definition(), "completed_updates": self.completed, "last_applied_rate": self.last_rate,
                "applied_rates_float64_le_sha256": self.digest.hexdigest(), "trace": self.trace}


def require_schedule(evidence, steps):
    digest = hashlib.sha256()
    trace = []
    for update in range(1, steps + 1):
        rate = learning_rate(update)
        digest.update(struct.pack("<d", rate))
        if update in (1, 2, HOLD_STEPS + 1) or update % 1000 == 0:
            trace.append({"update": update, "learning_rate": rate})
    expected = {**schedule_definition(), "completed_updates": steps, "last_applied_rate": learning_rate(steps),
                "applied_rates_float64_le_sha256": digest.hexdigest(), "trace": trace}
    if evidence != expected:
        raise ValueError("actual successful-update LR trace differs from registered schedule")


def main():
    args = rms.parse_args()
    if (args.stage != "calibrate" or args.loss_kind != "rms_score" or args.steps not in (2, TOTAL_STEPS)
            or args.learning_rate != INITIAL_LR or list(args.train_components) != ["transmitter", "receiver"]
            or args.optimize_expert_index is not None or args.batch_size != 100):
        raise ValueError("V255 supports only the registered full72k run or initial two-step probe")
    if args.output_dir.exists():
        raise FileExistsError("V255 refuses implicit checkpoint/optimizer resume")
    original_adam, original_parse = torch.optim.Adam, rms.parse_args
    captured = []

    def adam(parameters, *, lr, **kwargs):
        if captured:
            raise RuntimeError("unexpected second optimizer construction")
        optimizer = original_adam(parameters, lr=lr, **kwargs)
        captured.append(RateRecorder(optimizer))
        return optimizer

    try:
        rms.parse_args = lambda: args
        torch.optim.Adam = adam
        rms.main()
    finally:
        torch.optim.Adam, rms.parse_args = original_adam, original_parse
    if len(captured) != 1:
        raise RuntimeError("registered Adam was not constructed")
    evidence = captured[0].evidence()
    require_schedule(evidence, args.steps)
    path = args.output_dir / "training_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["history"][-1]["step"] != args.steps:
        raise ValueError("full requested trajectory required before schedule annotation")
    report["learning_rate_schedule"] = evidence
    report["learning_rate_field_scope"] = "initial rate only; actual applied rates bound by learning_rate_schedule"
    for target in (path, args.output_dir.parent / "calibration.json", args.output_dir.parent / "latest_training.json"):
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"completed_schedule": evidence, "training_report": str(path)}), flush=True)


if __name__ == "__main__":
    main()

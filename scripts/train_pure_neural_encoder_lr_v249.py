"""Isolated two-rate Adam: Encoder 5e-5, Tx/Rx 1e-5, unchanged score objective."""
from __future__ import annotations

import json

import torch
import train_pure_neural_snr_experts as trainer


def parameter_groups(parameters, encoder_parameters, lr):
    parameters = list(parameters)
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    all_ids = {id(parameter) for parameter in parameters}
    if (len(all_ids) != len(parameters) or not encoder_ids or not encoder_ids < all_ids
            or any(not parameter.requires_grad for parameter in parameters)):
        raise ValueError("expected unique, trainable encoder and transceiver parameters")
    return [{"params": [p for p in parameters if id(p) in encoder_ids], "lr": lr * 5., "name": "encoder"},
            {"params": [p for p in parameters if id(p) not in encoder_ids], "lr": lr, "name": "transceiver"}]


def main():
    args = trainer.parse_args()
    if (args.stage != "calibrate" or list(args.train_components) != ["encoder", "transmitter", "receiver"]
            or args.learning_rate != 1e-5 or args.optimize_expert_index is not None):
        raise ValueError("V249 requires all-component calibration at base LR 1e-5")
    if args.output_dir.exists():
        raise FileExistsError("V249 refuses implicit checkpoint resume")
    original_parse = trainer.parse_args
    original_select = trainer.select_trainable_parameters
    original_adam = torch.optim.Adam
    captured = {}

    def select(link, components, expert_index):
        selected = original_select(link, components, expert_index)
        captured["encoder_parameters"] = list(link.encoder.parameters())
        return selected

    def adam(parameters, *, lr, **kwargs):
        if "groups" in captured:
            raise RuntimeError("unexpected second optimizer construction")
        groups = parameter_groups(parameters, captured["encoder_parameters"], lr)
        captured["groups"] = [{"name": group["name"], "lr": group["lr"],
                               "parameters": sum(p.numel() for p in group["params"])} for group in groups]
        return original_adam(groups, lr=lr, **kwargs)

    try:
        trainer.parse_args = lambda: args
        trainer.select_trainable_parameters = select
        torch.optim.Adam = adam
        trainer.main()
    finally:
        trainer.parse_args = original_parse
        trainer.select_trainable_parameters = original_select
        torch.optim.Adam = original_adam
    report_path = args.output_dir / "training_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.update(encoder_learning_rate=5e-5, transceiver_learning_rate=1e-5,
                  optimizer_parameter_groups=captured["groups"],
                  optimizer_group_note="single Adam; global gradient clipping unchanged; per-component LR only")
    for path in (report_path, args.output_dir.parent / "calibration.json", args.output_dir.parent / "latest_training.json"):
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

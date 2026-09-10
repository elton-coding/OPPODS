"""V260 real-parent routing identity, grouped gradients, and all-route Adam proof."""
from __future__ import annotations

import argparse
import gc
import io
import json
import time

import torch
from run_pure_neural_lr_v237 import ROOT, fingerprints
from run_pure_neural_pair_v253 import check_training_slot
from train_pure_neural_rms_v239 import rms_score_loss
from train_pure_neural_snr_experts import PureNeuralLink, load_model_design, select_trainable_parameters

from oppods.data import ChannelMemmap, deterministic_split_indices

DESIGN = "research/pure_neural_v260/modelDesign.py"
CONTROL_DESIGN = "research/pure_neural_v230/modelDesign.py"
PARENT = "artifacts/pure_neural_v227/joint_low"
MAPPING = [0, 0, 1, 1, 1, 1, 1, 1]
CPU_PROOF = "benchmarks/v260_cpu_receiver_routing_probe.json"
GPU_PROOF = "benchmarks/v260_full_state_gpu_probe.json"


def source_paths():
    names = [DESIGN, CONTROL_DESIGN, "scripts/probe_pure_neural_receiver_v260.py",
             "scripts/train_pure_neural_snr_experts.py", "scripts/train_pure_neural_rms_v239.py",
             "scripts/run_pure_neural_lr_v237.py", "scripts/run_pure_neural_pair_v253.py",
             "src/oppods/data.py", "ziliao/data_train/H_train.npz"]
    return [ROOT / n for n in names] + [ROOT / PARENT / f"{n}.pth" for n in ("encoder", "transmitter", "receiver")]


def objective(logits, bits):
    return rms_score_loss(logits, bits, loss_kind="rms_score", margin=.5, tail_weight=0.,
                          tail_fraction=.1, score_temperature=.5, quantile_bandwidth=.025,
                          score_bce_weight=.05, score_fairness_weight=.3, valid_lengths=None)


def grouped_gradients(link):
    """Only Rx gradients are summed within their identical initial parent groups."""
    result = {}
    for component in ("transmitter", "receiver"):
        for index, expert in enumerate(getattr(link, component).experts):
            destination = MAPPING[index] if component == "receiver" else index
            for name, parameter in expert.named_parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise ValueError("all routes must receive finite gradients")
                key = f"{component}.{destination}.{name}"
                if key in result:
                    result[key].add_(parameter.grad)
                else:
                    result[key] = parameter.grad.detach().clone()
    return result


def make_link(design, device):
    torch.manual_seed(15240)
    link = PureNeuralLink(load_model_design(ROOT / design))
    construction_rng = torch.get_rng_state().clone()
    link.initialize_from_expert_bank(ROOT / PARENT, MAPPING)
    select_trainable_parameters(link, ["transmitter", "receiver"])
    return link.to(device).train(), construction_rng


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    cuda = args.device == "cuda"
    output = ROOT / (GPU_PROOF if cuda else CPU_PROOF)
    if output.exists():
        raise FileExistsError("V260 proof exists; no overwrite or implicit restart")
    torch.set_num_threads(2)
    paths = source_paths()
    if cuda:
        check_training_slot()
        cpu = json.loads((ROOT / CPU_PROOF).read_text(encoding="utf-8"))
        if cpu.get("passed") is not True or cpu.get("input_sha256") != fingerprints(paths):
            raise ValueError("missing or changed CPU routing proof")
        paths.append(ROOT / CPU_PROOF)
        torch.cuda.set_per_process_memory_fraction(.4)
        torch.cuda.reset_peak_memory_stats()
    before, started = fingerprints(paths), time.perf_counter()
    device = torch.device(args.device)
    batch = 100 if cuda else 8
    data = ChannelMemmap(ROOT / "ziliao/data_train/H_train.npz")
    indices = deterministic_split_indices(len(data), seed=1176)["train"][:batch]
    channel = torch.from_numpy(data.read(indices)).to(device)
    bits = torch.randint(0, 2, (batch, 2, 1152), generator=torch.Generator().manual_seed(15260)).float().to(device)
    tx_routes = torch.arange(batch) % 8
    snr = torch.stack([-17.5 + 5. * tx_routes, torch.full((batch,), 20.)], dim=1).to(device)
    snr[1::2] = snr[1::2].flip(1)
    module = load_model_design(ROOT / DESIGN)
    assert module._expert_indices(snr.amin(1)).tolist() == tx_routes.tolist()
    control_bits = ((tx_routes[:, None] >> torch.arange(5)) & 1).float().to(device)
    rx_routes = torch.stack([module._receiver_indices(control_bits, snr[:, u]) for u in range(2)], 1)
    assert set(rx_routes.flatten().tolist()) == set(range(8))
    assert all(MAPPING[r] == MAPPING[t] for r, t in zip(
        rx_routes.flatten().tolist(), tx_routes.repeat_interleave(2).tolist(), strict=True))

    def singles(link):
        with torch.no_grad():
            return torch.cat([link(channel[i:i+1], bits[i:i+1], snr[i:i+1],
                                   generator=torch.Generator(device=device).manual_seed(17260+i)) for i in range(batch)])

    identity = None
    if not cuda:
        control, control_rng = make_link(CONTROL_DESIGN, device)
        control_singles = singles(control)
        generator = torch.Generator().manual_seed(16260)
        logits = control(channel, bits, snr, generator=generator)
        control_logits, noise_rng = logits.detach().clone(), generator.get_state().clone()
        loss = objective(logits, bits)
        control_loss = float(loss.detach())
        loss.backward()
        expected = grouped_gradients(control)
        del control, logits, loss
        gc.collect()
    candidate, construction_rng = make_link(DESIGN, device)
    parameters = select_trainable_parameters(candidate, ["transmitter", "receiver"])
    assert sum(p.numel() for p in candidate.parameters()) == 190354976
    assert sum(p.numel() for p in parameters) == 189551584
    assert len(parameters) == len({id(p) for p in parameters}) == 1296
    encoder_before = {n: p.detach().clone() for n, p in candidate.encoder.named_parameters()}
    if not cuda:
        assert torch.equal(control_rng, construction_rng)
        torch.testing.assert_close(singles(candidate), control_singles, atol=0, rtol=0)
        generator = torch.Generator().manual_seed(16260)
        logits = candidate(channel, bits, snr, generator=generator)
        torch.testing.assert_close(logits.detach(), control_logits, atol=2e-4, rtol=0)
        assert torch.equal(generator.get_state(), noise_rng)
        hard_changes = int(((logits.detach() > 0) != (control_logits > 0)).sum())
        assert hard_changes == 0
        logit_difference = float((logits.detach()-control_logits).abs().max())
        loss = objective(logits, bits)
        loss_difference = float(loss.detach())-control_loss
        assert abs(loss_difference) <= 1e-6
        loss.backward()
        actual = grouped_gradients(candidate)
        assert actual.keys() == expected.keys()
        differences = {"transmitter": 0., "receiver_parent_sum": 0.}
        for name, value in actual.items():
            torch.testing.assert_close(value, expected[name], atol=1e-5, rtol=1e-4)
            group = "transmitter" if name.startswith("transmitter.") else "receiver_parent_sum"
            differences[group] = max(differences[group], float((value-expected[name]).abs().max()))
        identity = {"single_logit_maximum_difference": 0., "batch_logit_maximum_difference": logit_difference,
                    "hard_decision_disagreements": hard_changes, "loss_difference": loss_difference,
                    "gradient_maximum_differences": differences, "construction_rng_equal": True,
                    "channel_rng_equal": True, "scope": "Tx per parameter; Rx summed per initial parent, not per expert"}
        del expected, actual, logits, loss, control_logits, control_singles
        gc.collect()
    optimizer = torch.optim.Adam(parameters, lr=1e-5)
    steps = []
    for step in (1, 2):
        optimizer.zero_grad(set_to_none=True)
        logits = candidate(channel, bits, snr, generator=torch.Generator(device=device).manual_seed(18260+step))
        loss = objective(logits, bits)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
        assert all(p.grad is None and not p.requires_grad for p in candidate.encoder.parameters())
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        optimizer.step()
        assert len(optimizer.state) == len(parameters)
        for p in parameters:
            assert torch.isfinite(p).all() and optimizer.state[p]["step"].item() == step
            assert all(not torch.is_tensor(v) or torch.isfinite(v).all() for v in optimizer.state[p].values())
        for name, p in candidate.encoder.named_parameters():
            torch.testing.assert_close(p, encoder_before[name], atol=0, rtol=0)
        steps.append({"step": step, "loss": float(loss.detach()), "gradient_norm_before_clip": float(norm),
                      "finite_gradient_tensors": len(parameters), "adam_parameter_states": len(optimizer.state)})
    del optimizer, logits, loss
    gc.collect()
    serialization = None
    if not cuda:
        with io.BytesIO() as stream:
            torch.save(candidate.state_dict(), stream)
            serialized_bytes = stream.tell()
            stream.seek(0)
            state = torch.load(stream, map_location="cpu", weights_only=True)
            restored = PureNeuralLink(module)
            restored.load_state_dict(state, strict=True)
        restored.train()
        torch.testing.assert_close(singles(restored), singles(candidate), atol=0, rtol=0)
        for name, value in candidate.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], value, atol=0, rtol=0)
        serialization = {"bytes": serialized_bytes, "all_parameters_and_outputs_equal": True}
    if cuda:
        torch.cuda.synchronize()
    if before != fingerprints(paths):
        raise RuntimeError("V260 proof inputs changed")
    record = {"purpose": "routing/gradient/runtime evidence, not competition score", "passed": True,
              "device": args.device, "input_sha256": before, "train_data_indices": indices.tolist(),
              "batch_size": batch, "snr": snr.tolist(), "mapping": MAPPING,
              "transmitter_route_counts": torch.bincount(tx_routes, minlength=8).tolist(),
              "receiver_route_counts": torch.bincount(rx_routes.flatten(), minlength=8).tolist(),
              "parameters": 190354976, "trainable_parameters": 189551584,
              "trainable_parameter_tensors": len(parameters), "identity": identity, "steps": steps,
              "encoder_frozen_and_unchanged": True, "serialization": serialization, "weights_saved": False,
              "elapsed_seconds": time.perf_counter()-started, "gpu_memory_fraction": .4 if cuda else None,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else 0,
              "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved() if cuda else 0}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps({k: v for k, v in record.items() if k != "input_sha256"}), flush=True)


if __name__ == "__main__":
    main()

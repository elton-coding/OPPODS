"""CPU-only identity, tied-gradient, optimizer and serialization proof for V257."""
from __future__ import annotations

import gc
import io
import json
import time

import torch
from run_pure_neural_lr_v237 import ROOT, fingerprints
from train_pure_neural_rms_v239 import rms_score_loss
from train_pure_neural_snr_experts import (
    PureNeuralLink,
    load_model_design,
    select_trainable_parameters,
)

from oppods.data import ChannelMemmap, deterministic_split_indices

DESIGN = "research/pure_neural_v257/modelDesign.py"
CONTROL_DESIGN = "research/pure_neural_v230/modelDesign.py"
PARENT = "artifacts/pure_neural_v227/joint_low"
MAPPING = [0, 0, 1, 1, 1, 1, 1, 1]
EVIDENCE = "benchmarks/v257_cpu_shared_prefix_probe.json"


def source_paths():
    names = [DESIGN, CONTROL_DESIGN, "scripts/probe_pure_neural_shared_v257.py",
             "scripts/train_pure_neural_snr_experts.py", "scripts/train_pure_neural_rms_v239.py",
             "src/oppods/data.py"]
    return [ROOT / name for name in names] + [ROOT / PARENT / f"{name}.pth"
            for name in ("encoder", "transmitter", "receiver")]


def shared_name(component, name):
    inputs = ("_bit_embed.", "_feedback_expand.", "_embed.") if component == "transmitter" else ("_embed.",)
    return name.startswith(inputs) or (name.startswith("_blocks.") and int(name.split(".")[1]) < 8)


def canonical_name(component, expert_index, name):
    if shared_name(component, name):
        expert_index = 0 if MAPPING[expert_index] == 0 else 2
    return f"{component}.experts.{expert_index}.{name}"


def assert_aliases(link):
    count = 0
    for component in ("transmitter", "receiver"):
        experts = getattr(link, component).experts
        for index, expert in enumerate(experts):
            leader = experts[0 if MAPPING[index] == 0 else 2]
            own, reference = dict(expert.named_parameters()), dict(leader.named_parameters())
            for name, parameter in own.items():
                if shared_name(component, name):
                    assert parameter is reference[name], (component, index, name)
                    count += 1
                elif expert is not leader:
                    assert parameter is not reference[name], (component, index, name)
            other = experts[2 if MAPPING[index] == 0 else 0]
            assert not ({id(p) for p in expert.parameters()} & {id(p) for p in other.parameters()})
    parameters = select_trainable_parameters(link, ["transmitter", "receiver"])
    assert len(parameters) == len({id(p) for p in parameters})
    assert not any(p.requires_grad for p in link.encoder.parameters())
    return count


def objective(logits, bits):
    return rms_score_loss(logits, bits, loss_kind="rms_score", margin=1., tail_weight=.3,
                          tail_fraction=.1, score_temperature=.5, quantile_bandwidth=.025,
                          score_bce_weight=.05, score_fairness_weight=.3)


def make_link(design):
    torch.manual_seed(15240)
    module = load_model_design(ROOT / design)
    if design == DESIGN and list(module.SHARED_PARENT_MAP) != MAPPING:
        raise ValueError("V257 sharing groups require the original V227 parent map")
    link = PureNeuralLink(module)
    construction_rng = torch.get_rng_state().clone()
    link.initialize_from_expert_bank(ROOT / PARENT, MAPPING)
    select_trainable_parameters(link, ["transmitter", "receiver"])
    return link.train(), construction_rng


def main():
    output = ROOT / EVIDENCE
    if output.exists():
        raise FileExistsError("CPU evidence exists; inspect it rather than overwrite")
    torch.set_num_threads(2)
    before, started = fingerprints(source_paths()), time.perf_counter()
    data = ChannelMemmap(ROOT / "ziliao/data_train/H_train.npz")
    indices = deterministic_split_indices(len(data), seed=1176)["train"][:8]
    channel = torch.from_numpy(data.read(indices))
    bits = torch.randint(0, 2, (8, 2, 1152), generator=torch.Generator().manual_seed(257)).float()
    snr = torch.stack([torch.arange(-17.5, 20., 5.), torch.full((8,), 20.)], dim=1)
    snr[1::2] = snr[1::2].flip(1)
    module = load_model_design(ROOT / DESIGN)
    assert module._expert_indices(snr.amin(1)).tolist() == list(range(8))

    def singles(link):
        with torch.no_grad():
            return torch.cat([link(channel[i:i+1], bits[i:i+1], snr[i:i+1],
                                   generator=torch.Generator().manual_seed(16257+i)) for i in range(8)])

    control, control_rng = make_link(CONTROL_DESIGN)
    control_singles = singles(control)
    generator = torch.Generator().manual_seed(15257)
    logits = control(channel, bits, snr, generator=generator)
    control_logits, noise_rng = logits.detach().clone(), generator.get_state().clone()
    loss = objective(logits, bits)
    control_loss = float(loss.detach())
    loss.backward()
    expected = {}
    for component in ("transmitter", "receiver"):
        for index, expert in enumerate(getattr(control, component).experts):
            for name, parameter in expert.named_parameters():
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                key = canonical_name(component, index, name)
                if key in expected:
                    expected[key].add_(parameter.grad)
                else:
                    expected[key] = parameter.grad.detach().clone()
    del control, logits, loss
    gc.collect()

    candidate, construction_rng = make_link(DESIGN)
    assert torch.equal(control_rng, construction_rng)
    alias_checks = assert_aliases(candidate)
    candidate_singles = singles(candidate)
    torch.testing.assert_close(candidate_singles, control_singles, atol=0, rtol=0)
    generator = torch.Generator().manual_seed(15257)
    logits = candidate(channel, bits, snr, generator=generator)
    torch.testing.assert_close(logits.detach(), control_logits, atol=0, rtol=0)
    assert torch.equal(generator.get_state(), noise_rng)
    loss = objective(logits, bits)
    assert float(loss.detach()) == control_loss
    loss.backward()
    actual = {name: parameter.grad for name, parameter in candidate.named_parameters() if parameter.requires_grad}
    assert actual.keys() == expected.keys()
    gradient_max = 0.
    for name, value in actual.items():
        assert value is not None and torch.isfinite(value).all()
        torch.testing.assert_close(value, expected[name], atol=1e-5, rtol=1e-4)
        gradient_max = max(gradient_max, float((value-expected[name]).abs().max()))
    del expected, actual, logits, loss
    gc.collect()

    parameters = select_trainable_parameters(candidate, ["transmitter", "receiver"])
    optimizer = torch.optim.Adam(parameters, lr=1e-5)
    steps = []
    for step in range(1, 3):
        optimizer.zero_grad(set_to_none=True)
        logits = candidate(channel, bits, snr, generator=torch.Generator().manual_seed(17257+step))
        loss = objective(logits, bits)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
        torch.nn.utils.clip_grad_norm_(parameters, 1.)
        optimizer.step()
        assert len(optimizer.state) == len(parameters)
        assert all(int(state["step"]) == step for state in optimizer.state.values())
        steps.append({"step": step, "adam_parameter_states": len(optimizer.state),
                      "finite_gradient_tensors": len(parameters)})
    assert_aliases(candidate)
    del optimizer, logits, loss
    gc.collect()

    # Tensor storage aliases must survive save/load and construction. No probe
    # checkpoint is written to a formal training path or reused for training.
    with io.BytesIO() as stream:
        torch.save(candidate.state_dict(), stream)
        serialized_bytes = stream.tell()
        stream.seek(0)
        state = torch.load(stream, map_location="cpu", weights_only=True)
        restored = PureNeuralLink(module)
        restored.load_state_dict(state, strict=True)
    assert_aliases(restored)
    restored.train()
    torch.testing.assert_close(singles(restored), singles(candidate), atol=0, rtol=0)
    for name, value in candidate.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, atol=0, rtol=0)
    if before != fingerprints(source_paths()):
        raise RuntimeError("probe inputs changed")
    record = {"purpose": "CPU identity/gradients/optimizer/serialization, not score evidence", "passed": True,
              "input_sha256": before, "train_data_indices": indices.tolist(), "split_seed": 1176,
              "snr": snr.tolist(), "routes": list(range(8)), "mapping": MAPPING,
              "shared_prefix_blocks": 8, "shared_parameter_alias_checks": alias_checks,
              "parameters": sum(p.numel() for p in candidate.parameters()),
              "trainable_parameters": sum(p.numel() for p in parameters),
              "trainable_parameter_tensors": len(parameters), "maximum_single_logit_difference": 0.,
              "maximum_batched_logit_difference": 0., "hard_decision_disagreements": 0,
              "loss_difference": 0., "maximum_tied_gradient_difference": gradient_max,
              "construction_rng_equal": True, "channel_rng_equal": True, "steps": steps,
              "serialization_bytes": serialized_bytes, "serialization_aliases_and_outputs_equal": True,
              "elapsed_seconds": time.perf_counter()-started}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()

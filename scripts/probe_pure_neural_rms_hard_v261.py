"""Real-model RMS hard-forward and manual surrogate-gradient runtime contract."""
from __future__ import annotations

import argparse
import json
import time

import torch
from run_pure_neural_lr_v237 import ROOT, fingerprints
from run_pure_neural_pair_v253 import check_training_slot
from train_pure_neural_rms_hard_rank_v261 import LOSS_KIND, MIN_RMS, rms_hard_rank_loss
from train_pure_neural_snr_experts import PureNeuralLink, load_model_design, select_trainable_parameters

from oppods.data import ChannelMemmap, deterministic_split_indices

DESIGN = "research/pure_neural_v230/modelDesign.py"
PARENT = "artifacts/pure_neural_v227/joint_low"
MAPPING = [0, 0, 1, 1, 1, 1, 1, 1]
CPU_PROOF = "benchmarks/v261_cpu_rms_hard_probe.json"
GPU_PROOF = "benchmarks/v261_full_state_gpu_probe.json"


def source_paths():
    names = [DESIGN, "scripts/probe_pure_neural_rms_hard_v261.py",
             "scripts/train_pure_neural_rms_hard_rank_v261.py", "scripts/train_pure_neural_snr_experts.py",
             "scripts/train_pure_neural_rms_v239.py", "scripts/run_pure_neural_lr_v237.py",
             "scripts/run_pure_neural_pair_v253.py", "src/oppods/data.py", "ziliao/data_train/H_train.npz"]
    return [ROOT / n for n in names] + [ROOT / PARENT / f"{n}.pth" for n in ("encoder", "transmitter", "receiver")]


def options(*, fairness=.3, bce=.05):
    return {"margin": .5, "tail_weight": 0., "tail_fraction": .1, "score_temperature": .5,
            "quantile_bandwidth": .025, "score_bce_weight": bce, "score_fairness_weight": fairness,
            "valid_lengths": None}


def objective(logits, bits, *, fairness=.3, bce=.05):
    return rms_hard_rank_loss(logits, bits, loss_kind=LOSS_KIND, **options(fairness=fairness,bce=bce))


def hard_forward_reference(logits, bits, *, fairness=.3, bce=.05):
    hard = ((logits.detach() >= 0) == (bits >= .5)).float().sum(-1).flatten()/1152
    signed = (2.*bits-1.)*logits
    return -((1.-fairness)*hard.mean()+fairness*torch.quantile(hard,.1)) + bce*torch.nn.functional.softplus(-signed).mean()


def manual_surrogate(logits, bits, *, fairness=.3, bce=.05):
    """Independent gradient reference, NOT an equal-forward-value loss."""
    hard = ((logits.detach() >= 0) == (bits >= .5)).to(logits.dtype).sum(-1).flatten()/1152
    order = torch.sort(hard).indices
    ranks = torch.arange(hard.numel(),device=logits.device,dtype=logits.dtype)
    weights = torch.softmax(-.5*((ranks-.1*(hard.numel()-1))/max(1.,.025*hard.numel())).square(),0)
    normalized = logits/logits.square().mean(-1,keepdim=True).clamp_min(MIN_RMS**2).sqrt()
    soft = torch.sigmoid((2.*bits-1.)*normalized/.5).sum(-1).flatten()/1152
    raw_bce = torch.nn.functional.softplus(-(2.*bits-1.)*logits).mean()
    return -((1.-fairness)*soft.mean()+fairness*(weights*soft[order]).sum())+bce*raw_bce


def contract(logits, bits):
    leaf = logits.detach().clone().requires_grad_(True)
    loss = objective(leaf,bits)
    expected = hard_forward_reference(leaf,bits)
    torch.testing.assert_close(loss.detach(),expected.detach(),atol=2e-7,rtol=0)
    gradient = torch.autograd.grad(loss,leaf)[0]
    reference = torch.autograd.grad(manual_surrogate(leaf,bits),leaf)[0]
    assert torch.isfinite(gradient).all() and torch.isfinite(reference).all()
    torch.testing.assert_close(gradient,reference,atol=1e-7,rtol=1e-5)
    return {"hard_forward_difference": float((loss-expected).detach()),
            "manual_surrogate_gradient_maximum_difference": float((gradient-reference).abs().max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    args = parser.parse_args()
    cuda = args.device == "cuda"
    output = ROOT / (GPU_PROOF if cuda else CPU_PROOF)
    if output.exists():
        raise FileExistsError("V261 runtime evidence exists; no overwrite/implicit rerun")
    torch.set_num_threads(2)
    paths = source_paths()
    if cuda:
        check_training_slot()
        cpu = json.loads((ROOT / CPU_PROOF).read_text(encoding="utf-8"))
        if cpu.get("passed") is not True or cpu.get("input_sha256") != fingerprints(paths):
            raise ValueError("missing or changed CPU hard-forward/gradient contract")
        paths.append(ROOT / CPU_PROOF)
        torch.cuda.set_per_process_memory_fraction(.4)
        torch.cuda.reset_peak_memory_stats()
    before, started = fingerprints(paths),time.perf_counter()
    device = torch.device(args.device)
    batch = 100 if cuda else 8
    data = ChannelMemmap(ROOT / "ziliao/data_train/H_train.npz")
    indices = deterministic_split_indices(len(data),seed=1176)["train"][:batch]
    channel = torch.from_numpy(data.read(indices)).to(device)
    bits = torch.randint(0,2,(batch,2,1152),generator=torch.Generator().manual_seed(15261)).float().to(device)
    routes = torch.arange(batch)%8
    snr = torch.stack([-17.5+5.*routes,torch.full((batch,),20.)],dim=1).to(device)
    snr[1::2] = snr[1::2].flip(1)
    module = load_model_design(ROOT / DESIGN)
    assert module._expert_indices(snr.amin(1)).tolist() == routes.tolist()
    torch.manual_seed(15240)
    link = PureNeuralLink(module)
    link.initialize_from_expert_bank(ROOT / PARENT,MAPPING)
    link = link.to(device).train()
    parameters = select_trainable_parameters(link,["transmitter","receiver"])
    assert sum(p.numel() for p in parameters) == 189551584
    assert sum(p.numel() for p in link.parameters()) == 190354976
    assert len(parameters) == len({id(p) for p in parameters}) == 1296
    encoder_before = {n:p.detach().clone() for n,p in link.encoder.named_parameters()}
    optimizer = torch.optim.Adam(parameters,lr=1e-5)
    steps = []
    for step in (1,2):
        optimizer.zero_grad(set_to_none=True)
        generator = torch.Generator(device=device).manual_seed(16260+step)
        logits = link(channel,bits,snr,generator=generator)
        noise_state = generator.get_state().clone()
        cpu_state = torch.get_rng_state().clone()
        cuda_state = torch.cuda.get_rng_state().clone() if cuda else None
        check = contract(logits,bits)
        loss = objective(logits,bits)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.equal(generator.get_state(),noise_state) and torch.equal(torch.get_rng_state(),cpu_state)
        if cuda:
            assert torch.equal(torch.cuda.get_rng_state(),cuda_state)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
        assert all(p.grad is None and not p.requires_grad for p in link.encoder.parameters())
        norm = torch.nn.utils.clip_grad_norm_(parameters,1.)
        optimizer.step()
        assert len(optimizer.state) == len(parameters)
        for p in parameters:
            assert torch.isfinite(p).all() and optimizer.state[p]["step"].item() == step
            assert all(not torch.is_tensor(v) or torch.isfinite(v).all() for v in optimizer.state[p].values())
        for n,p in link.encoder.named_parameters():
            torch.testing.assert_close(p,encoder_before[n],atol=0,rtol=0)
        steps.append({"step":step,"loss":float(loss.detach()),"gradient_norm_before_clip":float(norm),
                      "finite_gradient_tensors":len(parameters),"adam_parameter_states":len(optimizer.state),
                      "loss_backward_rng_unchanged":True,**check})
    if cuda:
        torch.cuda.synchronize()
    if before != fingerprints(paths):
        raise RuntimeError("V261 runtime proof sources changed")
    record = {"purpose":"hard-forward/surrogate-gradient/runtime evidence, not performance","passed":True,
              "device":args.device,"input_sha256":before,"batch_size":batch,"train_data_indices":indices.tolist(),
              "route_counts":torch.bincount(routes,minlength=8).tolist(),"mapping":MAPPING,"loss_kind":LOSS_KIND,
              "parameters":190354976,"trainable_parameters":189551584,"trainable_parameter_tensors":len(parameters),
              "steps":steps,"encoder_frozen_and_unchanged":True,"weights_saved":False,
              "elapsed_seconds":time.perf_counter()-started,"gpu_memory_fraction":.4 if cuda else None,
              "gpu_peak_allocated_bytes":torch.cuda.max_memory_allocated() if cuda else 0,
              "gpu_peak_reserved_bytes":torch.cuda.max_memory_reserved() if cuda else 0}
    with output.open("x",encoding="utf-8") as stream:
        json.dump(record,stream,indent=2)
    print(json.dumps({k:v for k,v in record.items() if k != "input_sha256"}),flush=True)


if __name__ == "__main__":
    main()

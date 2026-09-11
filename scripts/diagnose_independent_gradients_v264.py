"""Same-sample V250 native independent-prefix gradient control; no updates."""
from __future__ import annotations

import json

import torch
import diagnose_shared_gradients_v264 as shared
from audit_pure_neural_candidate import fingerprint
from probe_pure_neural_shared_v257 import shared_name
from run_pure_neural_lr_v237 import ROOT, fingerprints
from run_storage_factorial_v254 import verify_bound_inputs, write_new
from train_pure_neural_snr_experts import PureNeuralLink, load_model_design, sample_snr

from oppods.data import ChannelMemmap, deterministic_split_indices

SOURCE = ROOT / "artifacts/pure_neural_v250/eight/steps72000"
PLAN = ROOT / "benchmarks/v264_independent_gradient_plan.json"
OUTPUT = ROOT / "benchmarks/v264_independent_gradient_diagnostic.json"


def main():
    if PLAN.exists() or OUTPUT.exists():
        raise FileExistsError("diagnostic already exists; do not overwrite")
    prior = json.loads(shared.PLAN.read_text(encoding="utf-8"))
    verify_bound_inputs(prior)
    files = fingerprint(SOURCE)
    audit = ROOT / "benchmarks/v250_eight_rms_72k_audit_offset2000.json"
    if files != json.loads(audit.read_text(encoding="utf-8"))["files"]:
        raise ValueError("V250 files mismatch")
    paths = [ROOT / name for name in prior["input_sha256"]]
    paths += [shared.PLAN, shared.OUTPUT, audit, ROOT / "scripts/diagnose_independent_gradients_v264.py"]
    paths += [SOURCE / name for name in files]
    ids = deterministic_split_indices(100000, seed=1176)["train"][:400]
    if ids.tolist() != prior["data_indices"]:
        raise ValueError("training sample control mismatch")
    write_new(PLAN, {"input_sha256": fingerprints(paths), "data_indices": ids.tolist(), "seed": 26410,
        "batch_size": 100, "batches": 4, "optimizer_steps": 0,
        "parameter_space": "concatenated native independent prefixes of experts0 and1; no artificial tying",
        "caveat": "different trained checkpoints and parameterization; not a causal isolation of sharing"})
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.25)
    torch.manual_seed(26410)
    device = torch.device("cuda")
    link = PureNeuralLink(load_model_design(SOURCE / "modelDesign.py")).to(device).eval()
    link.load_submission(SOURCE)
    for p in link.parameters():
        p.requires_grad_(False)
    groups = {}
    for component in ("transmitter", "receiver"):
        groups[component] = [p for index in (0, 1)
            for name, p in getattr(link, component).experts[index].named_parameters() if shared_name(component, name)]
        for p in groups[component]:
            p.requires_grad_(True)
    parameters = groups["transmitter"] + groups["receiver"]
    assert len({id(p) for p in parameters}) == len(parameters)
    split_at = sum(p.numel() for p in groups["transmitter"])
    generator = torch.Generator(device=device).manual_seed(26410)
    data = ChannelMemmap(ROOT / "ziliao/data_train/H_train.npz")
    reference = json.loads(shared.OUTPUT.read_text(encoding="utf-8"))["records"]
    records = []
    for batch in range(4):
        channel = torch.from_numpy(data.read(ids[batch*100:(batch+1)*100])).to(device)
        bits = torch.randint(0, 2, (100, 2, 1152), generator=generator, device=device).float()
        snr = sample_snr(100, stage="calibrate", expert_index=None, device=device, generator=generator)
        logits = link(channel, bits, snr, generator=generator)
        parts = shared.contributions(logits, bits)
        loss = shared.rms_score_loss(logits, bits, loss_kind="rms_score", margin=.5, tail_weight=0.,
            tail_fraction=.1, score_temperature=.5, quantile_bandwidth=.025, score_bce_weight=.05, score_fairness_weight=.3)
        torch.testing.assert_close(parts.sum(), loss, atol=2e-7, rtol=1e-6)
        regions = shared.masks(snr)
        counts = {k: int(v.sum()) for k, v in regions.items()}
        assert counts == reference[batch]["counts"]
        vectors = {name: shared.gradient_vector(parts[mask].sum(), parameters) for name, mask in regions.items()}
        actual = shared.gradient_vector(loss, parameters)
        torch.testing.assert_close(sum(vectors.values()), actual, atol=1e-6, rtol=1e-4)
        assert all(torch.isfinite(v).all() for v in [actual, *vectors.values()])
        record = {"batch": batch, "counts": counts, "components": {},
                  "decomposition_max_error": float((sum(vectors.values())-actual).abs().max())}
        for component, index in (("tx", slice(0, split_at)), ("rx", slice(split_at, None)), ("joint", slice(None))):
            v = {name: value[index] for name, value in vectors.items()}
            record["components"][component] = {
                "norms": {name: float(value.norm()) for name, value in v.items()},
                "weak_strong_cosine": shared.cosine(v["weak_both_negative"], v["strong_with_weak_partner"]),
                "weak_total_cosine": shared.cosine(v["weak_both_negative"], actual[index])}
        records.append(record)
        print(json.dumps(record), flush=True)
        del channel, logits, bits, snr, parts, loss, vectors, actual
    verify_bound_inputs(json.loads(PLAN.read_text(encoding="utf-8")))
    assert fingerprint(SOURCE) == files and all(p.grad is None for p in link.parameters())
    write_new(OUTPUT, {"records": records, "model_files": files, "parameters_diagnosed": sum(p.numel() for p in parameters),
        "optimizer_steps": 0, "model_weights_saved": False, "plan": str(PLAN),
        "caveat": "Four training batches; native independent parameter space, no cross-model norm comparability or causal sharing attribution."})


if __name__ == "__main__":
    main()

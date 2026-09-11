"""Package and CPU-check frozen V268; deliberately not a release or score claim."""
from __future__ import annotations

import hashlib
import io
import itertools
import json
import zipfile

import torch
from audit_pure_neural_candidate import ROOT, fingerprint
from build_submission import ARCHIVE_MEMBERS
from evaluate_submission import load_model_design
from probe_pure_neural_shared_v257 import assert_aliases
from run_frozen_confirmation_v268 import PLAN as PLAN_PATH
from run_storage_factorial_v254 import verify_bound_inputs
from run_pure_neural_lr_v237 import fingerprints
from run_storage_factorial_v254 import write_new
from train_pure_neural_snr_experts import PureNeuralLink

DIRECTORY = ROOT / "artifacts/pure_neural_v268/eight/lr2e5_steps72000"
ARCHIVE = ROOT / "artifacts/pure_neural_v268/candidate_package/v268_shared_lr_candidate_not_released.zip"
OUTPUT = ROOT / "benchmarks/v268_candidate_package_verification.json"


def verify_archive(path, files):
    if path.stat().st_size > 1_000_000_000:
        raise ValueError("archive exceeds decimal 1GB limit")
    members = {}
    with zipfile.ZipFile(path) as archive:
        if tuple(archive.namelist()) != ARCHIVE_MEMBERS or archive.testzip() is not None:
            raise ValueError("archive layout or CRC failed")
        for name in ARCHIVE_MEMBERS:
            with archive.open(name) as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            record = {"sha256": digest, "bytes": archive.getinfo(name).file_size}
            if record != files[name.rsplit("/", 1)[1]]:
                raise ValueError(f"archive member differs from frozen source: {name}")
            members[name] = record
    return members


def snr_cases():
    cases = [(float(s), float(s)) for s in range(-20, 21, 5)]
    cases += [(float(a), float(b)) for a, b in itertools.product((-20, -10, 0, 20), repeat=2) if a != b]
    cases += [(-19., -1.), (-1., -19.), (6., 19.), (19., 6.)]
    return cases


def make_plan():
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    verify_bound_inputs(plan)
    return plan


def main():
    if OUTPUT.exists() or ARCHIVE.exists():
        raise FileExistsError("candidate package was already attempted; preserve and inspect it")
    registered = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if make_plan() != registered:
        raise RuntimeError("confirmation freeze changed before candidate packaging")
    files = fingerprint(DIRECTORY)
    if files != registered["files"][1]:
        raise RuntimeError("candidate differs from frozen confirmation source")
    sources = fingerprints([ROOT / name for name in (
        "scripts/verify_candidate_package_v268.py", "scripts/build_submission.py",
        "scripts/probe_pure_neural_shared_v257.py", "scripts/evaluate_submission.py",
        "scripts/train_pure_neural_snr_experts.py", "benchmarks/v268_confirmation_plan.json")])
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in ARCHIVE_MEMBERS:
            archive.write(DIRECTORY / name.rsplit("/", 1)[1], arcname=name)
    members = verify_archive(ARCHIVE, files)
    torch.set_num_threads(2)
    torch.manual_seed(257)
    module = load_model_design(DIRECTORY / "modelDesign.py")
    link = PureNeuralLink(module).cpu().eval()
    keys_checked = 0
    # Read actual archived weights, not just the original directory. Every alias
    # key must round-trip; silently overwriting inconsistent aliases is rejected.
    with zipfile.ZipFile(ARCHIVE) as archive:
        for component in ("encoder", "transmitter", "receiver"):
            state = torch.load(io.BytesIO(archive.read(f"submit_pt/modelSubmit/{component}.pth")),
                               map_location="cpu", weights_only=True)
            model = getattr(link, component)
            model.load_state_dict(state, strict=True)
            restored = model.state_dict()
            if state.keys() != restored.keys() or any(not torch.equal(value, restored[key])
                                                      for key, value in state.items()):
                raise RuntimeError("archived state did not restore exactly, including shared aliases")
            keys_checked += len(state)
            del state, restored
    aliases = assert_aliases(link)
    parameters = sum(p.numel() for p in link.parameters())
    if aliases != 1024 or parameters != 73547840:
        raise ValueError("shared architecture or effective capacity changed after archive loading")
    routes, outputs = set(), 0
    with torch.inference_mode():
        for case in snr_cases():
            channel = torch.randn(1, 2, 2, 16, 144, dtype=torch.complex64)
            bits = [torch.randint(0, 2, (1, 1152)).float() for _ in range(2)]
            snr = torch.tensor(case).reshape(2, 1)
            feedback = [link.encoder(channel[:, u], snr[u]) for u in range(2)]
            assert all(v.shape == (1, 96) and v.dtype == torch.complex64 and torch.isfinite(v).all()
                       for v in feedback)
            signal, control = link.transmitter(bits, feedback, snr)
            assert signal.shape == (1, 16, 144) and signal.dtype == torch.complex64
            assert torch.isfinite(signal).all()
            assert control.shape == (1, 5) and torch.all(control == control.square())
            route = int((control.long() * (2 ** torch.arange(5))).sum())
            assert route == min(7, max(0, int((min(case) + 20) // 5)))
            routes.add(route)
            energy = signal.abs().square().sum(dim=1).mean()
            assert torch.isfinite(energy) and energy > 0
            signal = signal / energy.sqrt()
            for user in range(2):
                received = (channel[:, user] * signal.unsqueeze(1)).sum(dim=2)
                received += torch.randn_like(received) * (10 ** (-case[user] / 20))
                llr = link.receiver(received, channel[:, user], control, snr[user])
                assert llr.shape == (1, 1152) and llr.dtype == torch.float32 and torch.isfinite(llr).all()
                outputs += 1
    assert len(snr_cases()) == 25 and outputs == 50 and routes == set(range(8))
    assert assert_aliases(link) == 1024
    if fingerprint(DIRECTORY) != files or make_plan() != registered:
        raise RuntimeError("frozen sources changed during package check")
    if sources != fingerprints([ROOT / name for name in sources]):
        raise RuntimeError("package verifier dependencies changed")
    with ARCHIVE.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    result = {"archive": str(ARCHIVE.relative_to(ROOT)), "archive_bytes": ARCHIVE.stat().st_size,
              "archive_sha256": digest, "members": members, "crc_passed": True, "files": files,
              "input_sha256": sources, "cpu_contract_snr_pairs": snr_cases(), "ue_outputs_checked": outputs,
              "routes_checked": sorted(routes), "archive_state_keys_exactly_restored": keys_checked,
              "shared_parameter_alias_checks_after_archive_loading": aliases, "total_parameters": parameters,
              "torch": torch.__version__, "device": "cpu", "released": False,
              "organizer_inference_time_certified": False, "online_confirmation": False,
              "confirmation_result_claimed": False,
              "scope": "candidate-only ZIP and synthetic CPU contract; not score, promotion, or organizer timing"}
    write_new(OUTPUT, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

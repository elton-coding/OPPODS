"""Verify the exact frozen V250 archive and CPU interface; not organizer timing."""
from __future__ import annotations

import hashlib
import itertools
import json
import zipfile

import torch
from audit_pure_neural_candidate import ROOT, fingerprint
from build_submission import ARCHIVE_MEMBERS
from evaluate_submission import load_model_design
from run_frozen_confirmation_v250 import PLAN_PATH, decision, make_plan

DIRECTORY = ROOT / "artifacts/pure_neural_v250/eight/steps72000"
ARCHIVE = ROOT / "artifacts/FATE_MIMO_submission_pure_neural_v250_eight_rms_72k_official.zip"
OUTPUT = ROOT / "benchmarks/v250_release_verification.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError("release verification exists: inspect instead of overwrite")
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if make_plan() != plan:
        raise RuntimeError("confirmation freeze changed before release")
    confirm = json.loads((ROOT / "benchmarks/v250_confirm_audit_offset6000.json").read_text(encoding="utf-8"))
    if not decision(confirm)["supports_current_cohort_promotion"]:
        raise ValueError("confirmation failed")
    before = fingerprint(DIRECTORY)
    if before != confirm["files"] or ARCHIVE.stat().st_size > 1_000_000_000:
        raise ValueError("archive too large or source differs from confirmation")
    members = {}
    with zipfile.ZipFile(ARCHIVE) as archive:
        if tuple(archive.namelist()) != ARCHIVE_MEMBERS or archive.testzip() is not None:
            raise ValueError("archive layout or CRC failed")
        for name in ARCHIVE_MEMBERS:
            with archive.open(name) as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            basename = name.rsplit("/", 1)[1]
            if digest != before[basename]["sha256"]:
                raise ValueError(f"archive member differs: {name}")
            members[name] = {"sha256": digest, "bytes": archive.getinfo(name).file_size}
    torch.set_num_threads(2)
    torch.manual_seed(250)
    module = load_model_design(DIRECTORY / "modelDesign.py")
    encoder, transmitter, receiver = module.Encoder().eval(), module.Transmitter().eval(), module.Receiver().eval()
    for model, name in ((encoder, "encoder"), (transmitter, "transmitter"), (receiver, "receiver")):
        model.load_state_dict(torch.load(DIRECTORY / f"{name}.pth", map_location="cpu", weights_only=True))
    cases = [(float(s), float(s)) for s in range(-20, 21, 5)]
    cases += [(float(a), float(b)) for a, b in itertools.product((-20, -10, 0, 20), repeat=2) if a != b]
    cases += [(-19., -1.), (-1., -19.), (6., 19.), (19., 6.)]
    routes, outputs = set(), 0
    with torch.inference_mode():
        for case in cases:
            channel = torch.randn(1, 2, 2, 16, 144, dtype=torch.complex64)
            bits = [torch.randint(0, 2, (1, 1152)).float() for _ in range(2)]
            snr = torch.tensor(case).reshape(2, 1)
            feedback = [encoder(channel[:, user], snr[user]) for user in range(2)]
            assert all(value.shape == (1, 96) and value.dtype == torch.complex64
                       and torch.isfinite(value).all() for value in feedback)
            signal, control = transmitter(bits, feedback, snr)
            assert signal.shape == (1, 16, 144) and signal.dtype == torch.complex64
            assert torch.isfinite(signal).all()
            assert control.shape == (1, 5) and torch.all(control == control.square())
            route = int((control.long() * (2 ** torch.arange(5))).sum())
            expected = min(7, max(0, int((min(case) + 20) // 5)))
            assert route == expected
            routes.add(route)
            energy = signal.abs().square().sum(dim=1).mean()
            assert torch.isfinite(energy) and energy > 0
            signal = signal / energy.sqrt()
            for user in range(2):
                received = (channel[:, user] * signal.unsqueeze(1)).sum(dim=2)
                received += torch.randn_like(received) * (10 ** (-case[user] / 20))
                llr = receiver(received, channel[:, user], control, snr[user])
                assert llr.shape == (1, 1152) and llr.dtype == torch.float32 and torch.isfinite(llr).all()
                outputs += 1
    assert len(cases) == 25 and outputs == 50 and routes == set(range(8))
    if fingerprint(DIRECTORY) != before or make_plan() != plan:
        raise RuntimeError("frozen files changed during verification")
    with ARCHIVE.open("rb") as stream:
        archive_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    result = {"archive": str(ARCHIVE.relative_to(ROOT)), "archive_bytes": ARCHIVE.stat().st_size,
              "archive_sha256": archive_hash, "members": members, "crc_passed": True,
              "files": before, "cpu_contract_snr_pairs": cases, "ue_outputs_checked": outputs,
              "routes_checked": sorted(routes), "torch": torch.__version__,
              "organizer_inference_time_certified": False, "online_confirmation": False,
              "scope": "synthetic CPU contract plus frozen archive identity; not score or organizer timing"}
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

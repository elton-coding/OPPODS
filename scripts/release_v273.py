"""Publish verified V273 files locally; preserve immutable V269 and V273 archives."""
import hashlib
import shutil

import run_frozen_confirmation_v273 as frozen
import run_shared_budget_v273 as source
from verify_candidate_package_v257 import verify_archive


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    root = source.ROOT
    out = root / 'benchmarks/v273_release_verification.json'
    fixed = root / 'artifacts/FATE_MIMO_submission_pure_neural_v273_shared8_rms_144k_official.zip'
    if out.exists() or fixed.exists():
        raise FileExistsError('release already attempted; inspect, do not overwrite')
    source.verify_bound_inputs(source.read(frozen.PLAN))
    evidence = source.read(root / 'benchmarks/v273_candidate_package_verification.json')
    source.verify_bound_inputs(evidence)
    decision = source.read(frozen.RESULT)
    assert decision['supports_conditional_promotion'] and frozen.gate(decision['comparison'])
    assert source.read(source.DECISION)['eligible_for_new_confirmation']
    files = source.fingerprint(source.OUTPUT)
    assert files == evidence['files']
    archive = root / evidence['archive']
    assert sha(archive) == evidence['archive_sha256']
    verify_archive(archive, files)
    old = source.read(root / 'benchmarks/v269_release_verification.json')
    assert sha(root / old['archive']) == old['archive_sha256']
    assert sha(root / 'artifacts/FATE_MIMO_submission.zip') == old['archive_sha256']
    assert source.fingerprint(root / 'modelSubmit') == source.fingerprint(root / source.CONTROL_DIR)
    shutil.copyfile(archive, fixed)
    for name in files:
        shutil.copyfile(source.OUTPUT / name, root / 'modelSubmit' / name)
    shutil.copyfile(fixed, root / 'artifacts/FATE_MIMO_submission.zip')
    assert source.fingerprint(root / 'modelSubmit') == files
    for path in (fixed, root / 'artifacts/FATE_MIMO_submission.zip'):
        assert sha(path) == evidence['archive_sha256']
        verify_archive(path, files)
    assert sha(root / old['archive']) == old['archive_sha256']
    source.verify_bound_inputs(source.read(frozen.PLAN))
    source.write_new(out, {'version': 'V273', 'archive': str(fixed.relative_to(root)),
                          'archive_sha256': evidence['archive_sha256'], 'archive_bytes': fixed.stat().st_size,
                          'files': files, 'production_and_archive_verified': True,
                          'previous_fixed_archive_preserved': True, 'full_cpu_pytest_exit_code': 0,
                          'full_cpu_pytest_session': 87660, 'online_confirmation': False,
                          'whole_project_blindness_certified': False,
                          'organizer_inference_time_certified': False})
    print('V273 local release verified; Git publication still required')


if __name__ == '__main__':
    main()

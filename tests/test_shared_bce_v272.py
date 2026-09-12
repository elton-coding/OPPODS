import copy

import pytest
import run_shared_bce_v272 as v


def test_only_bce_and_output():
    for probe in (False, True):
        cmd, base = v.command(probe), v.baseline_command(probe=probe)
        assert cmd[cmd.index('--score-bce-weight')+1] == '0.025'
        assert cmd[cmd.index('--learning-rate')+1] == '3e-5'
        for flag in ('--score-bce-weight', '--output-dir'):
            i = base.index(flag)+1
            cmd[i] = base[i]
        assert cmd == base
    assert v.CONTROL == 'v269_shared8_rms_lr3e5_72k'


def test_guard():
    for probe, path in ((True, 'artifacts/resource_probe/v269/shared_lr/training_report.json'),
                        (False, 'artifacts/pure_neural_v269/eight/lr3e5_steps72000/training_report.json')):
        report = copy.deepcopy(v.read(v.ROOT / path))
        report['score_bce_weight'] = 0.025
        v.require_result(report, probe)
        for key, value in [('learning_rate', 3.5e-5), ('learning_rate_schedule', {}),
                           ('trainable_parameters', 83767368), ('score_bce_weight', 0.05),
                           ('requested_steps', 1), ('microbatch_size', 10)]:
            with pytest.raises(ValueError):
                v.require_result({**report, key: value}, probe)
        report['history'] = report['history'][:-1]
        with pytest.raises(ValueError):
            v.require_result(report, probe)

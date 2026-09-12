import copy
import hashlib
import struct

import pytest
import run_shared_cosine_v275 as v
import train_pure_neural_cosine_v275 as s


def evidence(steps):
    rates = [s.learning_rate(i) for i in range(1, steps+1)]
    return {**s.schedule_definition(), 'completed_updates': steps,
            'last_applied_rate': rates[-1],
            'applied_rates_float64_le_sha256': hashlib.sha256(b''.join(struct.pack('<d', x) for x in rates)).hexdigest(),
            'trace': [{'update': i, 'learning_rate': r} for i, r in enumerate(rates, 1)
                      if i in (1, 2, 72001) or i % 1000 == 0]}


def test_schedule_endpoints_and_command():
    assert s.learning_rate(72000) == 3e-5
    assert s.learning_rate(144000) == pytest.approx(1e-5)
    assert 1e-5 < s.learning_rate(108000) < 3e-5
    for probe in (False, True):
        cmd, base = v.command(probe), v.baseline_command(probe=probe)
        assert cmd[2] == 'scripts/train_pure_neural_cosine_v275.py'
        cmd[2] = base[2]
        cmd[cmd.index('--output-dir')+1] = base[base.index('--output-dir')+1]
        assert cmd == base


def test_full_report_and_prefix_guard():
    base = v.read(v.ROOT / v.CONTROL_DIR / 'training_report.json')
    full = copy.deepcopy(base)
    full['learning_rate_schedule'] = evidence(144000)
    full['learning_rate_field_scope'] = 'initial rate only; actual applied rates bound by learning_rate_schedule'
    v.require_result(full)
    assert v.check_prefix(full, base)['checkpoints'] == 73
    for key, value in [('learning_rate', 4e-5), ('score_bce_weight', .025),
                       ('requested_steps', 72000), ('learning_rate_schedule', evidence(2))]:
        with pytest.raises(ValueError):
            v.require_result({**full, key: value})
    full['history'][50]['final'] += .1
    with pytest.raises(ValueError):
        v.check_prefix(full, base)

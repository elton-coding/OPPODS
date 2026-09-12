import copy

import pytest
import run_shared_budget_v273 as v


def test_budget_command():
    for probe in (False, True):
        cmd, base = v.command(probe), v.baseline_command(probe=probe)
        assert cmd[cmd.index('--steps')+1] == ('2' if probe else '144000')
        for flag in ('--steps', '--patience', '--output-dir'):
            i = base.index(flag)+1
            cmd[i] = base[i]
        assert cmd == base


def test_budget_guard_and_prefix():
    short = v.read(v.ROOT / 'artifacts/pure_neural_v269/eight/lr3e5_steps72000/training_report.json')
    with pytest.raises(ValueError):
        v.require_result(short)
    full = copy.deepcopy(short)
    full['requested_steps'] = 144000
    full['history'] += [{**full['history'][-1], 'step': i} for i in range(73000, 144001, 1000)]
    v.require_result(full)
    assert v.check_prefix(full, short)['checkpoints'] == 73
    for field, value in [('learning_rate', 4e-5), ('score_bce_weight', 0.025),
                         ('requested_steps', 143000), ('trainable_parameters', 1)]:
        with pytest.raises(ValueError):
            v.require_result({**full, field: value})
    full['history'][-1]['final'] = float('nan')
    with pytest.raises(ValueError):
        v.require_result(full)
    full['history'][10]['fairness'] += 0.1
    with pytest.raises(ValueError):
        v.check_prefix(full, short)

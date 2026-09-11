import copy

import pytest
from run_frozen_confirmation_v268 import SEEDS, gate


def test_confirmation_gate():
    good = {'per_seed': [{'noise_seed': s, 'delta': {'final': 0.1}} for s in SEEDS],
            'mean_delta': 0.1, 'paired_delta_95_percentile_interval': [0.01, 0.2]}
    assert gate(good)
    bad = copy.deepcopy(good)
    bad['per_seed'][1]['delta']['final'] = -0.01
    assert not gate(bad)
    bad = copy.deepcopy(good)
    bad['paired_delta_95_percentile_interval'][0] = -0.01
    assert not gate(bad)
    bad = copy.deepcopy(good)
    bad['per_seed'][0]['noise_seed'] = 22701
    with pytest.raises(ValueError):
        gate(bad)
    bad = copy.deepcopy(good)
    bad['mean_delta'] = float('nan')
    with pytest.raises(ValueError):
        gate(bad)

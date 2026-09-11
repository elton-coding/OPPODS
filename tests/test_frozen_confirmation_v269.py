import copy

import pytest
from run_frozen_confirmation_v269 import SEEDS, require_metrics


def test_actual_confirmation_protocol():
    rows = [{'seed': s, 'samples': 2000, 'scores': 4000, 'short_outputs': 0,
             'min_output_length': 1152, 'max_output_length': 1152, 'split_seed': 1176,
             'test_offset': 8000, 'final': 68., 'efficiency': 71., 'fairness': 61.} for s in SEEDS]
    report = {'protocol': {'split_seed': 1176, 'test_offset': 8000, 'samples': 2000,
                           'noise_seeds': SEEDS}, 'per_seed': rows, 'exact_mean_final': 68.}
    cached = [{'final': 68., 'efficiency': 71., 'p10': 61.}] * 3
    require_metrics(report, cached)
    for field, value in [('seed', 22701), ('min_output_length', 576), ('final', 69.),
                         ('test_offset', 2000)]:
        bad = copy.deepcopy(report)
        bad['per_seed'][0][field] = value
        with pytest.raises(ValueError):
            require_metrics(bad, cached)

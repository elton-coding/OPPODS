"""Read-only recovery after all six evaluations; preserve original failure and bindings."""
import math

import run_frozen_confirmation_v268 as frozen
import run_shared_lr_v268 as source


def require_metrics(report, cached):
    if report['protocol'] != {'split_seed': 1176, 'test_offset': 8000, 'samples': 2000,
                              'noise_seeds': frozen.SEEDS}:
        raise ValueError('wrong confirmation protocol')
    if [r['seed'] for r in report['per_seed']] != frozen.SEEDS:
        raise ValueError('wrong noise seeds')
    for row, actual in zip(report['per_seed'], cached, strict=True):
        if any(row[k] != v for k, v in {'samples': 2000, 'scores': 4000, 'short_outputs': 0,
                                      'min_output_length': 1152, 'max_output_length': 1152,
                                      'split_seed': 1176, 'test_offset': 8000}.items()):
            raise ValueError('incomplete full payload')
        for key, mapped in [('final', 'final'), ('efficiency', 'efficiency'), ('fairness', 'p10')]:
            if not math.isfinite(row[key]) or abs(row[key] - actual[mapped]) > 1e-4:
                raise ValueError('cached metric mismatch')
    if abs(report['exact_mean_final'] - sum(r['final'] for r in report['per_seed']) / 3) > 1e-10:
        raise ValueError('mean mismatch')


def main():
    if frozen.RESULT.exists():
        raise FileExistsError('decision already exists')
    assert source.read(frozen.FAILURE)['error'] == "ValueError('wrong full-payload audit protocol')"
    plan = source.read(frozen.PLAN)
    source.verify_bound_inputs(plan)
    assert source.data_record() == plan['dataset']
    caches = source.checked_caches(frozen.LABELS, offset=frozen.OFFSET, seeds=frozen.SEEDS)
    reports = []
    for i, label in enumerate(frozen.LABELS):
        report = source.read(source.ROOT / f'benchmarks/{label}_audit_offset8000.json')
        assert report['label'] == label
        assert report['files'] == plan['files'][i] == source.fingerprint(frozen.DIRS[i])
        assert report['training_report'] == source.read(frozen.DIRS[i] / 'training_report.json')
        (source.baseline_guard if i == 0 else source.require_result)(report['training_report'])
        require_metrics(report, caches[label])
        reports.append(report)
    comp = source.compare(*[source.score_paths(source.ROOT / 'benchmarks', label, frozen.SEEDS, 8000)
                            for label in frozen.LABELS])
    assert comp == reports[1]['comparisons'][frozen.LABELS[0]]
    result = {'supports_conditional_promotion': frozen.gate(comp), 'comparison': comp,
              'exact_mean_final': reports[1]['exact_mean_final'],
              'historical_channel_overlap': '2000/2000', 'whole_project_blindness_certified': False,
              'ancestor_training_provenance_certified': False, 'automatic_promotion': False,
              'online_confirmation': False, 'recovery': 'cache-only; original failure retained',
              'verified_inputs': len(plan['input_sha256'])}
    source.verify_bound_inputs(plan)
    source.write_new(frozen.RESULT, result)
    print({k: v for k, v in result.items() if k != 'comparison'})
    print({k: comp[k] for k in ['mean_delta', 'paired_delta_95_percentile_interval']})
    print([r['delta'] for r in comp['per_seed']])


if __name__ == '__main__':
    main()

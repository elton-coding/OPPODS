"""One frozen conditional confirmation; all channels historically reused."""
from __future__ import annotations

import math
import subprocess
import sys

import run_shared_cosine_v275 as source

ROOT = source.ROOT
SEEDS = [92701, 92702, 92703]
OFFSET = 8000
LABELS = ['v273_confirm_v275', 'v275_confirm']
DIRS = [ROOT / source.CONTROL_DIR, source.OUTPUT]
PLAN = ROOT / 'benchmarks/v275_confirmation_plan.json'
RESULT = ROOT / 'benchmarks/v275_confirmation_decision.json'
FAILURE = ROOT / 'benchmarks/v275_confirmation_failure.json'


def gate(comp):
    rows = comp['per_seed']
    interval = comp['paired_delta_95_percentile_interval']
    if [r['noise_seed'] for r in rows] != SEEDS or len(interval) != 2:
        raise ValueError('wrong confirmation protocol')
    values = [comp['mean_delta'], *interval, *[r['delta']['final'] for r in rows]]
    if not all(math.isfinite(x) for x in values) or interval[0] > interval[1]:
        raise ValueError('invalid comparison')
    return min(r['delta']['final'] for r in rows) > 0 and interval[0] > 0


def require_metrics(report, cached):
    if report['protocol'] != {'split_seed': 1176, 'test_offset': 8000, 'samples': 2000,
                              'noise_seeds': SEEDS}:
        raise ValueError('wrong confirmation protocol')
    if [r['seed'] for r in report['per_seed']] != SEEDS:
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
    audits = [ROOT / f'benchmarks/{label}_audit_offset{OFFSET}.json' for label in LABELS]
    paths = [p for label in LABELS for p in source.score_paths(ROOT / 'benchmarks', label, SEEDS, OFFSET)]
    if any(p.exists() for p in [PLAN, RESULT, FAILURE, *audits, *paths]):
        raise FileExistsError('reserved confirmation outputs exist; no restart')
    source.verify_bound_inputs(source.read(source.PLAN))
    candidate = source.read(ROOT / f'benchmarks/{source.LABEL}_audit_offset2000.json')
    control = source.read(ROOT / f'benchmarks/{source.CONTROL}_audit_offset2000.json')
    source.require_result(candidate['training_report'])
    source.baseline_guard(control['training_report'])
    for directory, audit in zip(DIRS, [control, candidate], strict=True):
        if source.fingerprint(directory) != audit['files']:
            raise ValueError('source weights changed')
    binding = [ROOT / p for p in source.read(source.PLAN)['input_sha256']]
    binding += [source.PLAN, source.DECISION, *[d / name for d in DIRS
                for name in ('modelDesign.py', 'encoder.pth', 'transmitter.pth', 'receiver.pth', 'training_report.json')]]
    binding += [ROOT / p for p in ('scripts/run_frozen_confirmation_v275.py',
                'tests/test_frozen_confirmation_v275.py', 'docs/experiments/frozen-confirmation-v275.md',
                'benchmarks/v275_confirmation_channel_inventory.json')]
    plan = {'input_sha256': source.fingerprints(binding), 'dataset': source.data_record(),
            'protocol': {'split_seed': 1176, 'test_offset': OFFSET, 'samples': 2000, 'noise_seeds': SEEDS},
            'labels': LABELS, 'files': [control['files'], candidate['files']],
            'historical_channel_overlap': '2000/2000', 'whole_project_blindness_certified': False,
            'automatic_promotion': False}
    source.write_new(PLAN, plan)
    try:
        source.wait_gpu_slot()
        for i, (label, directory) in enumerate(zip(LABELS, DIRS, strict=True)):
            source.verify_bound_inputs(plan)
            subprocess.run([sys.executable, '-u', 'scripts/audit_pure_neural_candidate.py',
                            '--submission', str(directory), '--label', label,
                            '--baseline-label', LABELS[0] if i else '',
                            '--test-offset', str(OFFSET), '--samples', '2000',
                            '--noise-seeds', *map(str, SEEDS)], cwd=ROOT, check=True)
        caches = source.checked_caches(LABELS, offset=OFFSET, seeds=SEEDS)
        reports = [source.read(p) for p in audits]
        for i, report in enumerate(reports):
            assert report['protocol'] == plan['protocol'] and report['label'] == LABELS[i]
            assert report['files'] == plan['files'][i] == source.fingerprint(DIRS[i])
            assert report['training_report'] == source.read(DIRS[i] / 'training_report.json')
            require_metrics(report, caches[LABELS[i]])
        comp = source.compare(source.score_paths(ROOT / 'benchmarks', LABELS[0], SEEDS, OFFSET),
                              source.score_paths(ROOT / 'benchmarks', LABELS[1], SEEDS, OFFSET))
        assert comp == reports[1]['comparisons'][LABELS[0]]
        source.verify_bound_inputs(plan)
        assert source.data_record() == plan['dataset']
        result = {'supports_conditional_promotion': gate(comp), 'comparison': comp,
                  'exact_mean_final': reports[1]['exact_mean_final'],
                  'historical_channel_overlap': '2000/2000', 'whole_project_blindness_certified': False,
                  'ancestor_training_provenance_certified': False, 'automatic_promotion': False,
                  'online_confirmation': False}
        source.write_new(RESULT, result)
        print(result, flush=True)
    except Exception as error:
        source.write_new(FAILURE, {'error': repr(error), 'partial_outputs_preserved': True})
        raise


if __name__ == '__main__':
    main()

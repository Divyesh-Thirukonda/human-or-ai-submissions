"""Run actual-data audit, ten core candidates, five learning sizes and optional structural ablations."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import pandas as pd

from data_utils import audit_data, get_or_create_split, load_data, nested_stratified_subsets
from experiment import THREADS, evaluate, paired_bootstrap, write_json
from models import SEED


def candidate_name(config):
    key = 'alpha' if config['kind'] == 'word_nb' else 'C'
    value = f'{config[key]:g}'.replace('.', 'p')
    return f'{config["kind"]}_{key}{value}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--results', type=Path, default=Path('results'))
    parser.add_argument('--bonus', action='store_true', help='Four prespecified style/POS ablations after core experiments')
    parser.add_argument('--resume', action='store_true', help='Verify and reuse completed runs with identical data/configuration')
    args = parser.parse_args()
    out = args.results
    if not args.resume and (out / 'comparison.json').exists():
        parser.error('Comparison results already exist; choose a new results directory or --resume')
    data = load_data(args.data / 'train.csv', expected_rows=10000)
    train, validation = get_or_create_split(data, out / 'split.json', seed=SEED)
    other = {}
    for key, name, labeled, rows in [('unlabeled', 'unlabeled.csv', False, 3000),
                                    ('test', 'test.csv', False, 3000),
                                    ('starter', 'train_starter_2000.csv', True, 2000)]:
        path = args.data / name
        other[key] = load_data(path, labeled=labeled, expected_rows=rows) if path.is_file() else None
    audit = audit_data(data, train, validation, **other)
    write_json(out / 'audit.json', audit)
    sensitivity = audit.get('source_overlap_sensitivity', {}).get('excluded_validation_ids')
    environment = {'python': platform.python_version(), 'platform': platform.platform(),
                   'versions': {name: importlib.metadata.version(name) for name in
                                ('numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'joblib', 'nltk')},
                   'seed': SEED, 'threads': THREADS, 'schema': {'id': 'id', 'text': 'text', 'label': 'label'},
                   'source_sha256': data.attrs['source_sha256'], 'NLTK_DATA': os.environ.get('NLTK_DATA', 'not set'),
                   'calibration': 'Three seeded stratified inner folds; sigmoid; training-only shared vocabulary/IDF'}
    if not (args.resume and (out / 'environment.json').is_file()):
        write_json(out / 'environment.json', environment)
    candidates = [{'kind': kind, 'C': c} for kind in ('word_lr', 'char_lr') for c in (.3, 1., 3.)]
    candidates += [{'kind': 'word_nb', 'alpha': alpha} for alpha in (.1, 1., 3.)]
    candidates += [{'kind': 'word_svm', 'C': 1.}]
    plan = {'candidates': candidates, 'selection': 'maximum full-validation AUROC; exact ties first listed',
            'learning_sizes': [500, 1000, 2000, 5000, 8000], 'seed': SEED,
            'hypothesis_scope': 'Data-grounded lexical baselines; structural suggestions are separate exploratory ablations.'}
    plan_path = out / 'search_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        parser.error('Existing search plan differs; choose a new results directory')
    write_json(plan_path, plan)
    runs = []
    for config in candidates:
        runs.append(evaluate(config, train, validation, out / 'runs', candidate_name(config),
                             save_model=True, resume=args.resume, sensitivity_exclude_ids=sensitivity))
    best = max(runs, key=lambda row: row['validation_auroc'])
    write_json(out / 'comparison.json', runs)
    write_json(out / 'selected_config.json', best['config'])
    write_json(out / 'selection.json', {'selected': best,
               'scope': 'Day 1 core comparison; bonus results do not silently replace this selection'})
    curves = []
    for size, subset in nested_stratified_subsets(train, seed=SEED).items():
        write_json(out / 'learning_ids' / f'{size}.json', subset.id.astype(str).tolist())
        curves.append(evaluate(best['config'], subset, validation, out / 'learning', f'n{size}',
                               resume=args.resume, sensitivity_exclude_ids=sensitivity))
    write_json(out / 'learning_curve.json', curves)
    if args.bonus:
        configs = [{'kind': kind, 'C': 1.} for kind in ('style_lr', 'pos_lr', 'surprisal_lr', 'word_style_lr')]
        write_json(out / 'bonus_plan.json', {'candidates': configs, 'baseline': {'kind': 'word_lr', 'C': 1.},
                   'fixed': 'same 8000/2000 split, C=1; no bonus tuning',
                   'timing_note': 'POS cache is shared per process; later runs may reuse tagging work.'})
        bonus_runs, failures = [], []
        for config in configs:
            try:
                bonus_runs.append(evaluate(config, train, validation, out / 'bonus', config['kind'],
                                           save_model=True, resume=args.resume, sensitivity_exclude_ids=sensitivity))
            except Exception as exc:
                failure = {'config': config, 'type': type(exc).__name__, 'error': str(exc)}
                failures.append(failure)
                write_json(out / 'bonus' / f'{config["kind"]}.failure.json', failure)
                print(f'BONUS FAILED: {config["kind"]}: {exc}', flush=True)
        baseline_name = candidate_name({'kind': 'word_lr', 'C': 1.})
        baseline = pd.read_csv(out / 'runs' / f'{baseline_name}.predictions.tsv', sep='\t', dtype={'id': str})
        effects = {}
        for run in bonus_runs:
            prediction = pd.read_csv(out / 'bonus' / f'{run["name"]}.predictions.tsv', sep='\t', dtype={'id': str})
            if baseline.id.tolist() != prediction.id.tolist() or baseline.label.tolist() != prediction.label.tolist():
                raise ValueError('Bonus bootstrap inputs are not paired by identical IDs and labels')
            effects[run['name']] = paired_bootstrap(baseline.label, baseline.probability_ai, prediction.probability_ai)
        write_json(out / 'bonus_comparison.json', {'runs': bonus_runs, 'failures': failures, 'effects_vs_word_C1': effects})
    print(f'Day 1 measured records written to {out}; selected {best["name"]} AUROC={best["validation_auroc"]:.6f}', flush=True)
    print('Generate the reviewed Q1/Q2 report from these records with the separate report command.', flush=True)


if __name__ == '__main__':
    main()

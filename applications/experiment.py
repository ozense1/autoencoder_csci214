"""Train-then-evaluate loops for hyperparameter sweeps and multi-seed runs.

Each run trains one autoencoder (reusing cached runs), registers it, and is
immediately evaluated, so runs never overwrite each other's results. Fixed
methods (baseline, Matryoshka) are evaluated once. Results are written to
artifacts/experiments/<split>/<name>.json and .csv, with a mean ± std summary.

Development sweep (relevance labels select hyperparameters, never train):
    python applications/experiment.py --name gamma-sweep --split dev \
        --methods mse_cosine_geometry --gamma 0 0.1 1 10 --seeds 1 --split-seed 7

Final multi-seed comparison, only after settings are frozen:
    python applications/experiment.py --name final --split test \
        --methods mse mse_cosine mse_cosine_geometry --gamma 1 --seeds 1 2 3 --split-seed 7
"""
import argparse
import csv
import itertools
import math
from pathlib import Path
import statistics

from config import AE_METHODS, ARTIFACT_DIR, FIXED_METHODS, MATRYOSHKA, QREL_SPLITS
from data_utils import load_dataset, resolve_resource, write_json
from embeddings import Embedder, embedding_spec
from matryoshka import build_matryoshka
from metrics import METRIC_NAMES
from retrieval import evaluate
from storage import load_registry, readiness
from training import train_variant


def label_for(method, gamma):
    return f'{method} (gamma={gamma:g})' if 'geometry' in method else method


def summarize(rows):
    """Group runs by label; report mean and sample std over seeds."""
    groups = {}
    for row in rows:
        groups.setdefault(row['label'], []).append(row)
    summary = []
    for label, runs in groups.items():
        item = {'label': label, 'dimension': runs[0]['dimension'], 'runs': len(runs)}
        for key in METRIC_NAMES:
            values = [run[key] for run in runs]
            item[key] = statistics.fmean(values)
            item[f'{key}_std'] = statistics.stdev(values) if len(values) > 1 else 0.
        summary.append(item)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--name', required=True, help='Experiment name used for the output files.')
    parser.add_argument('--split', choices=list(QREL_SPLITS), default='dev')
    parser.add_argument('--methods', nargs='+', choices=list(AE_METHODS), required=True)
    parser.add_argument('--fixed', nargs='*', choices=list(FIXED_METHODS), default=list(FIXED_METHODS),
                        help='Untrained reference methods to evaluate once (default: all).')
    parser.add_argument('--gamma', nargs='+', type=float, default=[1.], help='Swept only for geometry methods.')
    parser.add_argument('--seeds', nargs='+', type=int, default=[7], help='Initialization/minibatch seeds.')
    parser.add_argument('--split-seed', type=int, default=7, help='Fixed 90/10 document split for all runs.')
    parser.add_argument('--latent-dim', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--alpha', type=float, default=1.)
    parser.add_argument('--beta', type=float, default=.1)
    parser.add_argument('--lambda', dest='lam', type=float, default=.01)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--artifact-dir', type=Path, default=ARTIFACT_DIR)
    args = parser.parse_args()
    if any(not math.isfinite(g) or g < 0 for g in args.gamma):
        parser.error('--gamma values must be finite and nonnegative.')
    if args.split == 'test':
        print('*** TEST SPLIT: use only after the design and settings are frozen. ***')

    artifact_dir = args.artifact_dir.resolve()
    registry = load_registry(artifact_dir)
    baseline = registry.get('baseline')
    if not baseline:
        parser.error('Run `python applications/train_autoencoder.py prepare` first.')
    model_path = resolve_resource(artifact_dir, baseline['model_path'])
    data_dir = resolve_resource(artifact_dir, baseline['data_dir'])
    if embedding_spec(model_path, data_dir)[1] != baseline['base_fingerprint']:
        parser.error('Model, corpus, or runtime changed. Prepare matching artifacts first.')
    _, queries, qrels = load_dataset(data_dir, split=args.split)
    print(f'Split: {args.split} ({len(qrels)} queries)')

    engine = None
    def get_engine():
        nonlocal engine
        if engine is None:
            engine = Embedder(model_path, **baseline['runtime'])
        return engine

    rows = []
    def record(method, entry, report, seed=None, gamma=None):
        values = report['methods'][method]
        row = {'label': label_for(method, gamma) if gamma is not None else method, 'method': method,
               'seed': seed, 'gamma': gamma, 'dimension': values['dimension'], **values['metrics'],
                'best_epoch': entry.get('best_epoch'), 'epochs_completed': entry.get('epochs_completed'),
                'training_seconds': entry.get('training_seconds'), 'parameter_count': entry.get('parameter_count'),
                'index_fingerprint': entry['index_fingerprint'], 'report': report.get('report_path')}
        for key, value in (entry.get('latent_stats') or {}).items():
            row[f'latent_{key}'] = value
        rows.append(row)
        print(f"  {row['label']:<40} seed={seed}  nDCG@10={row['nDCG@10']:.4f}  R@100={row['Recall@100']:.4f}"
                + (f"  best_epoch={row['best_epoch']}" if row['best_epoch'] else ''))

    try:
        for method in args.fixed:
            if method in MATRYOSHKA:
                build_matryoshka(method, MATRYOSHKA[method], baseline, artifact_dir)
            entry = readiness(artifact_dir)[method]['entry']
            report = evaluate([method], {method: entry}, baseline, artifact_dir, queries, qrels, get_engine, split=args.split)
            record(method, entry, report)
        runs = []
        for method in args.methods:
            gammas = args.gamma if 'geometry' in method else [None]
            runs += [(method, gamma, seed) for gamma, seed in itertools.product(gammas, args.seeds)]
        for number, (method, gamma, seed) in enumerate(runs, 1):
            print(f'[{number}/{len(runs)}] {label_for(method, gamma)} seed={seed}')
            entry = train_variant(method, baseline, artifact_dir, latent_dim=args.latent_dim, epochs=args.epochs,
                                    batch_size=args.batch_size, lr=args.lr, alpha=args.alpha, beta=args.beta,
                                    lam=args.lam, gamma=1. if gamma is None else gamma, seed=seed,
                                    split_seed=args.split_seed, patience=args.patience, threads=args.threads)
            report = evaluate([method], {method: entry}, baseline, artifact_dir, queries, qrels, get_engine, split=args.split)
            record(method, entry, report, seed, gamma)
    finally:
        if engine:
            engine.close()

    summary = summarize(rows)
    folder = artifact_dir / 'experiments' / args.split
    write_json(folder / f'{args.name}.json', {'split': args.split, 'arguments': {k: str(v) for k, v in vars(args).items()},
                                                'runs': rows, 'summary': summary})
    with (folder / f'{args.name}.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nSummary ({args.split}, mean ± std over seeds)')
    print(f"{'Method':<40}{'Dim':>5}{'n':>3}  " + '  '.join(f'{m:>17}' for m in METRIC_NAMES))
    for item in summary:
        cells = '  '.join(f"{item[m]:.4f} ± {item[m + '_std']:.4f}" for m in METRIC_NAMES)
        print(f"{item['label']:<40}{item['dimension']:>5}{item['runs']:>3}  {cells}")
    print('Saved:', folder / f'{args.name}.json')


if __name__ == '__main__':
    main()

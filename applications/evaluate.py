"""Evaluate prepared methods on SciFact queries without opening the UI.

--split test (default): the 300 held-out queries in qrels/test.tsv.
    Run only after the design and hyperparameters are frozen.
--split dev: the 809 queries in qrels/train.tsv, used for relevance-based model selection. 
    Reports go to artifacts/evaluations/dev/.
"""
import argparse
from pathlib import Path

from config import ARTIFACT_DIR, METHODS, QREL_SPLITS
from data_utils import load_dataset, resolve_resource
from embeddings import Embedder, embedding_spec
from retrieval import evaluate
from storage import load_registry, readiness


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--artifact-dir', type=Path, default=ARTIFACT_DIR)
    parser.add_argument('--methods', nargs='+', choices=list(METHODS), default=None,
                        help='Default: every prepared method.')
    parser.add_argument('--split', choices=list(QREL_SPLITS), default='test')
    parser.add_argument('--model-path', type=Path, default=None)
    args = parser.parse_args()
    available = readiness(args.artifact_dir)
    methods = args.methods or [m for m in METHODS if available.get(m, {}).get('ready')]
    missing = [m for m in methods if not available.get(m, {}).get('ready')]
    if missing or not methods:
        parser.error(f'Prepare/train these methods first: {missing or "all"}')
    baseline = load_registry(args.artifact_dir)['baseline']
    model_path = args.model_path or resolve_resource(args.artifact_dir, baseline['model_path'])
    data_dir = resolve_resource(args.artifact_dir, baseline['data_dir'])
    _, actual = embedding_spec(model_path, data_dir)
    if actual != baseline['base_fingerprint']:
        parser.error('Model, corpus, or runtime changed. Prepare matching artifacts first.')
    _, queries, qrels = load_dataset(data_dir, split=args.split)
    print(f'Split: {args.split} ({len(qrels)} queries, qrels/{QREL_SPLITS[args.split]})')
    engine = None
    def get_engine():
        nonlocal engine
        if engine is None:
            engine = Embedder(model_path, **baseline['runtime'])
        return engine
    try:
        report = evaluate(methods, {k: v['entry'] for k, v in available.items() if v['ready']}, baseline,
                            args.artifact_dir, queries, qrels, get_engine, split=args.split,
                            progress=lambda done, total: print(f'Evaluated {done}/{total}', flush=True) if done % 25 == 0 else None)
        for method, values in report['methods'].items():
            print(method, {k: round(v, 4) for k, v in values['metrics'].items()})
        print('Saved:', report['report_path'])
    finally:
        if engine:
            engine.close()


if __name__ == '__main__':
    main()

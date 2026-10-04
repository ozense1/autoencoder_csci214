"""Evaluate prepared methods on every SciFact test query without opening the UI."""
import argparse
from pathlib import Path

from config import ARTIFACT_DIR, METHODS
from data_utils import load_dataset, resolve_resource
from embeddings import Embedder, embedding_spec
from retrieval import evaluate
from storage import load_registry, readiness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir', type=Path, default=ARTIFACT_DIR)
    parser.add_argument('--methods', nargs='+', choices=list(METHODS), default=list(METHODS))
    parser.add_argument('--model-path', type=Path, default=None)
    args = parser.parse_args()
    available = readiness(args.artifact_dir)
    if any(not available.get(method, {}).get('ready') for method in args.methods):
        parser.error('Prepare/train every requested method first.')
    baseline = load_registry(args.artifact_dir)['baseline']
    model_path = args.model_path or resolve_resource(args.artifact_dir, baseline['model_path'])
    data_dir = resolve_resource(args.artifact_dir, baseline['data_dir'])
    _, actual = embedding_spec(model_path, data_dir)
    if actual != baseline['base_fingerprint']:
        parser.error('Model, corpus, or runtime changed. Prepare matching artifacts first.')
    _, queries, qrels = load_dataset(data_dir)
    engine = None
    def get_engine():
        nonlocal engine
        if engine is None:
            engine = Embedder(model_path, **baseline['runtime'])
        return engine
    try:
        report = evaluate(args.methods, {k: v['entry'] for k, v in available.items() if v['ready']}, baseline,
                          args.artifact_dir, queries, qrels, get_engine,
                          progress=lambda done, total: print(f'Evaluated {done}/{total}', flush=True) if done % 25 == 0 else None)
        for method, values in report['methods'].items():
            print(method, values['metrics'])
        print('Saved:', args.artifact_dir / 'evaluations' / f"{report['evaluation_fingerprint']}.json")
    finally:
        if engine:
            engine.close()


if __name__ == '__main__':
    main()

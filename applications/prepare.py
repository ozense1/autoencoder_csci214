"""Resumable document embedding preparation, called from the training CLI."""
from pathlib import Path
import time

import numpy as np

from config import DIMENSION
from data_utils import load_dataset, portable_path, read_json, write_json
from embeddings import Embedder, document_prompt, embedding_spec
from storage import build_collection, load_registry, save_registry


def prepare(model_path, data_dir, artifact_dir, threads=4, gpu_layers=0, progress=print):
    artifact_dir = Path(artifact_dir)
    documents, _, _ = load_dataset(data_dir)
    spec, base_fp = embedding_spec(model_path, data_dir)
    directory = artifact_dir / 'embeddings' / base_fp[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'documents.npy'
    manifest_path = directory / 'manifest.json'
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest['fingerprint'] != base_fp or manifest['document_count'] != len(documents):
            raise ValueError('Embedding cache fingerprint mismatch.')
        vectors = np.lib.format.open_memmap(path, mode='r+')
    else:
        vectors = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(len(documents), DIMENSION))
        manifest = {'fingerprint': base_fp, 'spec': spec, 'completed': 0, 'document_count': len(documents), 'truncated_documents': 0}
        write_json(manifest_path, manifest)
        write_json(directory / 'ids.json', [str(d['_id']) for d in documents])
    if vectors.shape != (len(documents), DIMENSION):
        raise ValueError('Embedding cache has an unexpected shape.')
    start = manifest['completed']
    if start < len(documents):
        embedder = Embedder(model_path, threads=threads, gpu_layers=gpu_layers)
        begun = time.perf_counter()
        try:
            for i in range(start, len(documents)):
                vector, truncated = embedder.encode(document_prompt(documents[i]))
                vectors[i] = vector
                manifest['truncated_documents'] += int(truncated)
                if (i + 1) % 32 == 0 or i + 1 == len(documents):
                    vectors.flush()
                    manifest['completed'] = i + 1
                    write_json(manifest_path, manifest)
                    progress(f'Embedded {i + 1}/{len(documents)} documents; elapsed {time.perf_counter() - begun:.0f}s')
        finally:
            embedder.close()
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1., atol=1e-4):
        raise ValueError('Invalid vectors in completed embedding cache.')
    collection = f'scifact-baseline-{base_fp[:16]}'
    registry = load_registry(artifact_dir)
    old = registry.get('baseline')
    # A complete registry allows repeat prepare calls to avoid rebuilding the index.
    from storage import readiness
    available = readiness(artifact_dir).get('baseline', {})
    if not (old and old['base_fingerprint'] == base_fp and available.get('ready')):
        build_collection(artifact_dir, collection, documents, vectors, base_fp, progress)
    entry = {
        'base_fingerprint': base_fp, 'index_fingerprint': base_fp,
        'collection': collection, 'dimension': DIMENSION, 'document_count': len(documents),
        'model_path': portable_path(model_path, artifact_dir),
        'data_dir': portable_path(data_dir, artifact_dir),
        'embedding_file': str(path.relative_to(artifact_dir)), 'spec': spec,
        'truncated_documents': manifest['truncated_documents'],
        'runtime': {'threads': threads, 'gpu_layers': gpu_layers},
    }
    if not old or old['base_fingerprint'] != base_fp:
        registry['variants'] = {}
    registry['baseline'] = entry
    save_registry(artifact_dir, registry)
    progress(f'Baseline ready: {len(documents)} documents, {DIMENSION} dimensions.')
    return entry

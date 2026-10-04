"""Matryoshka truncation baseline: first k coordinates, then L2-normalize again.

EmbeddingGemma is trained with Matryoshka Representation Learning, so its model
card documents truncating the 768-dimensional output to 512, 256 or 128 and
renormalizing. The identical transform is applied to queries and documents.
Nothing is trained; this is a fixed compression baseline at matched dimension.
"""
import numpy as np

from data_utils import fingerprint, load_dataset, resolve_resource
from embeddings import normalize_rows
from storage import build_collection, load_registry, readiness, save_registry


def truncate(vectors, dimension):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError('Expected a [samples, dimensions] array.')
    if not 0 < dimension < vectors.shape[1]:
        raise ValueError(f'Matryoshka dimension must be in 1..{vectors.shape[1] - 1}; got {dimension}.')
    return normalize_rows(vectors[:, :dimension])


def build_matryoshka(method, dimension, baseline, artifact_dir, progress=print):
    """Index truncated document vectors and register the method; reuse if ready."""
    index_fp = fingerprint({'base': baseline['base_fingerprint'], 'transform': 'matryoshka-prefix-v1',
                            'dimension': dimension, 'normalize': True})
    current = readiness(artifact_dir).get(method, {})
    if current.get('ready') and current['entry'].get('index_fingerprint') == index_fp:
        progress(f'{method}: reusing existing index.')
        return current['entry']
    vectors = np.load(resolve_resource(artifact_dir, baseline['embedding_file']))
    latent = truncate(vectors, dimension)
    documents, _, _ = load_dataset(resolve_resource(artifact_dir, baseline['data_dir']))
    collection = f'scifact-{method.replace("_", "-")}-{index_fp[:16]}'
    build_collection(artifact_dir, collection, documents, latent, index_fp, progress)
    entry = {
        'base_fingerprint': baseline['base_fingerprint'], 'index_fingerprint': index_fp,
        'collection': collection, 'dimension': dimension, 'document_count': len(documents),
        'transform': {'type': 'matryoshka', 'dimension': dimension},
    }
    registry = load_registry(artifact_dir)
    registry['variants'][method] = entry
    save_registry(artifact_dir, registry)
    progress(f'{method}: ready at {dimension} dimensions.')
    return entry

"""Explicit-vector Chroma collections and artifact readiness checks."""
from pathlib import Path
import chromadb
from chromadb.config import Settings
import numpy as np

from data_utils import read_json, sha256_file, write_json


def client(artifact_dir):
    return chromadb.PersistentClient(
        path=str(Path(artifact_dir) / 'chroma'),
        settings=Settings(anonymized_telemetry=False),
    )


def load_registry(artifact_dir):
    path = Path(artifact_dir) / 'current.json'
    return read_json(path) if path.exists() else {'baseline': None, 'variants': {}}


def save_registry(artifact_dir, registry):
    write_json(Path(artifact_dir) / 'current.json', registry)


def build_collection(artifact_dir, name, documents, vectors, fingerprint, progress=print):
    vectors = np.asarray(vectors, dtype=np.float32)
    if len(documents) != len(vectors):
        raise ValueError('Document/vector count mismatch.')
    db = client(artifact_dir)
    collection = db.get_or_create_collection(
        name=name, embedding_function=None,
        metadata={'fingerprint': fingerprint, 'dimensions': vectors.shape[1]},
        configuration={'hnsw': {'space': 'cosine', 'ef_construction': 200, 'ef_search': 200, 'max_neighbors': 16, 'num_threads': 4}},
    )
    if collection.metadata.get('fingerprint') != fingerprint:
        raise ValueError('Collection fingerprint mismatch; refusing to mix vector spaces.')
    for start in range(0, len(documents), 256):
        rows = documents[start:start + 256]
        collection.upsert(
            ids=[str(row['_id']) for row in rows],
            embeddings=vectors[start:start + len(rows)].tolist(),
            documents=[row['text'] for row in rows],
            metadatas=[{'title': row.get('title', ''), 'document_id': str(row['_id'])} for row in rows],
        )
    if collection.count() != len(documents):
        raise ValueError('Incomplete Chroma collection.')
    progress(f'Indexed {collection.count()} documents in {name}')
    return collection


def readiness(artifact_dir):
    registry = load_registry(artifact_dir)
    result = {}
    baseline = registry.get('baseline')
    entries = {'baseline': baseline, **registry.get('variants', {})}
    db = client(artifact_dir) if baseline else None
    for method, entry in entries.items():
        if not entry:
            continue
        try:
            if not baseline or entry['base_fingerprint'] != baseline['base_fingerprint']:
                raise ValueError('Embeddings changed; retrain this variant.')
            trained = 'checkpoint' in entry  # baseline and Matryoshka have no checkpoint
            if method != 'baseline' and not trained and 'transform' not in entry:
                raise ValueError('Entry has neither a checkpoint nor a fixed transform.')
            if trained and not (Path(artifact_dir) / entry['checkpoint']).is_file():
                raise ValueError('Checkpoint missing.')
            if trained and sha256_file(Path(artifact_dir) / entry['checkpoint']) != entry['checkpoint_sha256']:
                raise ValueError('Checkpoint changed; rebuild this index.')
            collection = db.get_collection(entry['collection'], embedding_function=None)
            if collection.count() != entry['document_count']:
                raise ValueError('Index is incomplete.')
            if collection.metadata.get('fingerprint') != entry['index_fingerprint']:
                raise ValueError('Index belongs to a different model.')
            result[method] = {'ready': True, 'entry': entry, 'reason': ''}
        except Exception as exc:
            result[method] = {'ready': False, 'entry': entry, 'reason': str(exc)}
    return result

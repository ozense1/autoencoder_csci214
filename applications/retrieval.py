"""Query encoding, matched-space Chroma retrieval, and held-out evaluation."""
from pathlib import Path
import time
import threading

from filelock import FileLock
import numpy as np
import torch

from data_utils import fingerprint, read_json, sha256_file, write_json
from embeddings import normalize_rows, query_prompt
from matryoshka import truncate
from metrics import mean_metrics, query_metrics
from storage import client
from training import load_encoder


def query_embedding(query, baseline, artifact_dir, embedder):
    # Text and preprocessing/model fingerprint identify the reusable query vector.
    key = fingerprint({'query': query, 'base': baseline['base_fingerprint']})
    folder = Path(artifact_dir) / 'queries' / baseline['base_fingerprint'][:16]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{key}.npz'
    started = time.perf_counter()
    with FileLock(str(folder / f'{key}.lock')):
        cached = path.exists()
        if cached:
            with np.load(path, allow_pickle=False) as saved:
                vector, truncated = saved['vector'], bool(saved['truncated'])
        else:
            # Lazy callable means cached benchmark queries need no model load.
            engine = embedder() if callable(embedder) else embedder
            vector, truncated = engine.encode(query_prompt(query))
            temporary = folder / f'{key}.tmp.npz'
            np.savez(temporary, vector=vector, truncated=truncated)
            temporary.replace(path)
    vector = normalize_rows(vector[None])[0]
    if len(vector) != baseline['dimension']:
        raise ValueError('Query cache has the wrong embedding dimension.')
    return vector, {'seconds': time.perf_counter() - started, 'cached': cached, 'truncated': truncated}


class Retriever:
    def __init__(self, artifact_dir):
        self.artifact_dir = Path(artifact_dir)
        self.db = client(artifact_dir)
        self.encoders = {}
        self.lock = threading.RLock()

    def search(self, vector, method, entry, top_k=100):
        with self.lock:
            return self._search(vector, method, entry, top_k)

    def _search(self, vector, method, entry, top_k):
        started = time.perf_counter()
        if method == 'baseline':
            query = vector[None]
        elif entry.get('transform', {}).get('type') == 'matryoshka':
            # Same prefix-and-renormalize transform as the indexed documents.
            query = truncate(vector[None], entry['transform']['dimension'])
        else:
            key = entry['checkpoint_sha256']
            # Verify file content even if a previous encoder is cached in memory.
            if sha256_file(self.artifact_dir / entry['checkpoint']) != key:
                raise ValueError('Checkpoint changed; rerun training to rebuild the index.')
            if key not in self.encoders:
                self.encoders[key] = load_encoder(self.artifact_dir, entry)
            with torch.no_grad():
                query = self.encoders[key].encode(torch.from_numpy(vector.copy())[None]).numpy()
            query = normalize_rows(query)
        encoder_seconds = time.perf_counter() - started
        collection = self.db.get_collection(entry['collection'], embedding_function=None)
        if collection.metadata.get('fingerprint') != entry['index_fingerprint']:
            raise ValueError('Index and encoder do not match.')
        started = time.perf_counter()
        raw = collection.query(query_embeddings=query.tolist(), n_results=min(top_k, collection.count()), include=['documents', 'metadatas', 'distances'])
        search_seconds = time.perf_counter() - started
        rows = [
            {'id': doc_id, 'title': metadata.get('title', ''), 'text': document,
                'cosine': float(np.clip(1 - distance, -1, 1))}
            for doc_id, metadata, document, distance in zip(raw['ids'][0], raw['metadatas'][0], raw['documents'][0], raw['distances'][0])
        ]
        return rows, {'encoder_seconds': encoder_seconds, 'search_seconds': search_seconds}


def evaluate(methods, entries, baseline, artifact_dir, queries, qrels, embedder, progress=None, split='test'):
    retriever = Retriever(artifact_dir)
    results = {method: [] for method in methods}
    timings = {method: {'encoder_seconds': 0., 'search_seconds': 0.} for method in methods}
    per_query = {}
    query_seconds, query_cache_hits, truncated_queries = 0., 0, 0
    for number, query_id in enumerate(sorted(qrels), 1):
        vector, timing = query_embedding(queries[query_id], baseline, artifact_dir, embedder)
        query_seconds += timing['seconds']
        query_cache_hits += int(timing['cached'])
        truncated_queries += int(timing['truncated'])
        per_query[query_id] = {}
        for method in methods:
            rows, timing = retriever.search(vector, method, entries[method], top_k=100)
            metrics = query_metrics([row['id'] for row in rows], qrels[query_id])
            results[method].append(metrics)
            per_query[query_id][method] = {'metrics': metrics, 'ranking': [row['id'] for row in rows]}
            for key in timings[method]:
                timings[method][key] += timing[key]
        if progress:
            progress(number, len(qrels))
    evaluation = {
        'split': split, 'query_count': len(qrels), 'base_fingerprint': baseline['base_fingerprint'],
        'methods': {method: {'metrics': mean_metrics(results[method]), 'dimension': entries[method]['dimension'],
                                'index_fingerprint': entries[method]['index_fingerprint'],
                                **{key: entries[method][key] for key in ('config', 'best_epoch', 'latent_stats', 'training_seconds') if key in entries[method]},
                                'mean_encoder_seconds': timings[method]['encoder_seconds'] / len(qrels),
                                'mean_search_seconds': timings[method]['search_seconds'] / len(qrels)} for method in methods},
        'query_embedding_or_cache_seconds': query_seconds, 'query_cache_hits': query_cache_hits,
        'truncated_queries': truncated_queries, 'per_query': per_query,
        'judgments_fingerprint': fingerprint(qrels), 'queries_fingerprint': fingerprint(queries),
    }
    key = fingerprint({'base': baseline['base_fingerprint'], 'indexes': {m: entries[m]['index_fingerprint'] for m in methods},
                        'queries': evaluation['queries_fingerprint'], 'qrels': evaluation['judgments_fingerprint'], 'split': split})
    evaluation['evaluation_fingerprint'] = key
    # Test reports stay where the app looks for them; development reports are kept
    # in a separate folder so they can never be mistaken for final results.
    folder = Path(artifact_dir) / 'evaluations' / ('' if split == 'test' else split)
    write_json(folder / f'{key}.json', evaluation)
    evaluation['report_path'] = str(folder / f'{key}.json')
    return evaluation

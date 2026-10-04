"""Local GGUF inference. One text per call avoids the installed batch limitation."""
import importlib.metadata
import threading
from pathlib import Path

import numpy as np

from config import CONTEXT, DIMENSION, PROMPT_VERSION
from data_utils import fingerprint, sha256_file


def document_prompt(document):
    return f"title: {document.get('title') or 'none'} | text: {document['text']}"


def query_prompt(query):
    return f'task: search result | query: {query}'


def embedding_spec(model_path, data_dir):
    model_path = Path(model_path).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f'Embedding model not found: {model_path}')
    spec = {
        'model_sha256': sha256_file(model_path),
        'corpus_sha256': sha256_file(Path(data_dir) / 'corpus.jsonl'),
        'llama_cpp_version': importlib.metadata.version('llama-cpp-python'),
        'prompt_version': PROMPT_VERSION,
        'dimension': DIMENSION,
        'context_tokens': CONTEXT,
        'normalize': True,
        'truncation': 'first-2048-token-ids-including-special-tokens',
    }
    return spec, fingerprint(spec)


def normalize_rows(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError('Expected a finite [samples, dimensions] embedding array.')
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError('Zero or near-zero embeddings cannot be used for cosine retrieval.')
    return vectors / norms


class Embedder:
    def __init__(self, model_path, threads=4, gpu_layers=0):
        from llama_cpp import Llama
        self.lock = threading.RLock()
        self.model = Llama(
            model_path=str(model_path), embedding=True,
            n_ctx=CONTEXT, n_batch=CONTEXT, n_ubatch=CONTEXT,
            n_threads=threads, n_gpu_layers=gpu_layers, verbose=False,
        )
        if self.model.n_embd() != DIMENSION:
            raise ValueError(f'Expected {DIMENSION} embedding coordinates; got {self.model.n_embd()}.')

    def encode(self, prompt):
        with self.lock:
            tokens = self.model.tokenize(prompt.encode('utf-8'))
            truncated = len(tokens) > CONTEXT
            # The high-level runtime truncates this exact token stream to n_batch.
            vector = np.asarray(self.model.embed(prompt, normalize=True, truncate=True), dtype=np.float32)
        if vector.shape != (DIMENSION,):
            raise ValueError(f'Expected a pooled vector, got shape {vector.shape}.')
        return normalize_rows(vector[None, :])[0], truncated

    def close(self):
        with self.lock:
            self.model.close()

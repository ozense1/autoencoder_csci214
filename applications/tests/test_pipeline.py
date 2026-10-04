import json
from pathlib import Path

import numpy as np
import pytest
import torch

import prepare as preparation
from config import DATA_DIR
from data_utils import fingerprint, load_dataset, portable_path, read_json, resolve_resource, sha256_file, write_json
from embeddings import document_prompt, normalize_rows, query_prompt
from retrieval import Retriever, query_embedding
from storage import build_collection, client, load_registry, readiness, save_registry
from training import load_encoder, train_variant


def small_dataset(path, n=40):
    path.mkdir(parents=True, exist_ok=True)
    (path / 'qrels').mkdir(exist_ok=True)
    docs = [{'_id': str(i), 'title': f'Title {i}', 'text': f'Document {i}'} for i in range(n)]
    (path / 'corpus.jsonl').write_text('\n'.join(json.dumps(d) for d in docs) + '\n')
    (path / 'queries.jsonl').write_text(json.dumps({'_id': 'q', 'text': 'Test query'}) + '\n')
    (path / 'qrels' / 'test.tsv').write_text('query-id\tcorpus-id\tscore\nq\t0\t1\n')
    return docs


def test_dataset_copy_and_counts():
    documents, queries, qrels = load_dataset(DATA_DIR)
    assert (len(documents), len(queries), len(qrels)) == (5183, 1109, 300)
    manifest = read_json(DATA_DIR.parent / 'copy_manifest.json')
    for relative, metadata in manifest.items():
        assert sha256_file(DATA_DIR / relative) == metadata['sha256']


def test_prompt_formats_and_invalid_vectors():
    assert query_prompt('q') == 'task: search result | query: q'
    assert document_prompt({'title': '', 'text': 'body'}) == 'title: none | text: body'
    assert np.allclose(normalize_rows([[3, 4]]), [[.6, .8]])
    for invalid in ([[0, 0]], [[float('nan'), 1]], [1, 2]):
        with pytest.raises(ValueError):
            normalize_rows(invalid)


def test_chroma_persistence_and_isolation(tmp_path):
    docs = small_dataset(tmp_path / 'data', n=2)
    build_collection(tmp_path, 'space-one', docs, [[1, 0], [0, 1]], 'one', lambda _: None)
    build_collection(tmp_path, 'space-two', docs, [[0, 1], [1, 0]], 'two', lambda _: None)
    again = client(tmp_path)
    for name, expected in [('space-one', '0'), ('space-two', '1')]:
        collection = again.get_collection(name, embedding_function=None)
        assert collection.query(query_embeddings=[[1, 0]], n_results=1)['ids'][0] == [expected]
    with pytest.raises(ValueError, match='fingerprint'):
        build_collection(tmp_path, 'space-one', docs, [[1, 0], [0, 1]], 'changed')


def test_resume_after_interrupted_embedding_and_invalidate_config(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    small_dataset(data)
    calls = []
    should_fail = [True]
    class FakeEmbedder:
        def __init__(self, *args, **kwargs):
            pass
        def encode(self, prompt):
            i = int(prompt.rsplit(' ', 1)[-1])
            calls.append(i)
            if should_fail[0] and i == 35:
                raise RuntimeError('interrupted')
            return normalize_rows([[1., float(i), .5]])[0], i == 2
        def close(self):
            pass
    monkeypatch.setattr(preparation, 'DIMENSION', 3)
    monkeypatch.setattr(preparation, 'Embedder', FakeEmbedder)
    monkeypatch.setattr(preparation, 'embedding_spec', lambda *a: ({'mock': 1}, 'a' * 64))
    out = tmp_path / 'artifacts'
    with pytest.raises(RuntimeError, match='interrupted'):
        preparation.prepare('fake', data, out, progress=lambda _: None)
    manifest = next(out.glob('embeddings/*/manifest.json'))
    assert read_json(manifest)['completed'] == 32
    should_fail[0] = False
    calls.clear()
    entry = preparation.prepare('fake', data, out, progress=lambda _: None)
    assert calls == list(range(32, 40))
    assert entry['truncated_documents'] == 1
    assert readiness(out)['baseline']['ready']
    calls.clear()
    preparation.prepare('fake', data, out, progress=lambda _: None)
    assert calls == []
    registry = load_registry(out)
    registry['variants']['obsolete'] = {'base_fingerprint': 'old'}
    save_registry(out, registry)
    monkeypatch.setattr(preparation, 'embedding_spec', lambda *a: ({'mock': 2}, 'b' * 64))
    preparation.prepare('fake', data, out, progress=lambda _: None)
    assert load_registry(out)['variants'] == {}


def test_query_cache_separates_text_and_embedding_model(tmp_path):
    class Fake:
        calls = 0
        def encode(self, prompt):
            self.calls += 1
            return np.array([1., 0., 0.], dtype=np.float32), False
    engine = Fake()
    base = {'base_fingerprint': 'a' * 64, 'dimension': 3}
    _, first = query_embedding('query', base, tmp_path, engine)
    _, second = query_embedding('query', base, tmp_path, engine)
    assert not first['cached'] and second['cached'] and engine.calls == 1
    query_embedding('another', base, tmp_path, engine)
    query_embedding('query', {**base, 'base_fingerprint': 'b' * 64}, tmp_path, engine)
    assert engine.calls == 3


def test_training_reload_reuse_and_corruption_detection(tmp_path):
    data = tmp_path / 'data'
    docs = small_dataset(data)
    rng = np.random.default_rng(7)
    vectors = normalize_rows(rng.normal(size=(40, 8)).astype(np.float32))
    np.save(tmp_path / 'vectors.npy', vectors)
    base = {'base_fingerprint': 'a' * 64, 'dimension': 8, 'embedding_file': 'vectors.npy',
            'data_dir': portable_path(data, tmp_path), 'document_count': 40, 'collection': 'baseline-test', 'index_fingerprint': 'a' * 64}
    build_collection(tmp_path, base['collection'], docs, vectors, base['index_fingerprint'], lambda _: None)
    save_registry(tmp_path, {'baseline': base, 'variants': {}})
    entry = train_variant('mse_cosine_contractive', base, tmp_path, latent_dim=3, epochs=2, threads=1, progress=lambda _: None)
    model = load_encoder(tmp_path, entry)
    with torch.no_grad():
        z = normalize_rows(model.encode(torch.from_numpy(vectors)).numpy())
    stored = np.load((tmp_path / entry['checkpoint']).parent / 'document_latents.npy')
    np.testing.assert_allclose(z, stored)
    assert readiness(tmp_path)['mse_cosine_contractive']['ready']
    service = Retriever(tmp_path)
    rows, _ = service.search(vectors[0], 'mse_cosine_contractive', entry, top_k=5)
    assert len(rows) == 5
    reused = train_variant('mse_cosine_contractive', base, tmp_path, latent_dim=3, epochs=2, threads=1, progress=lambda _: None)
    assert reused['checkpoint_sha256'] == entry['checkpoint_sha256']
    (tmp_path / entry['checkpoint']).write_bytes(b'corrupted')
    assert not readiness(tmp_path)['mse_cosine_contractive']['ready']
    with pytest.raises(ValueError, match='changed'):
        load_encoder(tmp_path, entry)


def test_portable_resources_resolve_after_move_and_cwd_change(tmp_path, monkeypatch):
    import shutil
    original = tmp_path / 'original' / 'applications'
    (original / 'models').mkdir(parents=True)
    (original / 'models' / 'model.gguf').write_bytes(b'local-model')
    (original / 'data' / 'scifact').mkdir(parents=True)
    artifacts = original / 'artifacts'
    artifacts.mkdir()
    stored_model = portable_path(original / 'models' / 'model.gguf', artifacts)
    stored_data = portable_path(original / 'data' / 'scifact', artifacts)
    assert stored_model == '../models/model.gguf'
    assert stored_data == '../data/scifact'
    write_json(artifacts / 'current.json', {'baseline': {'model_path': stored_model, 'data_dir': stored_data}})
    moved = tmp_path / 'moved' / 'applications'
    shutil.copytree(original, moved)
    shutil.rmtree(original)
    monkeypatch.chdir(tmp_path)
    baseline = read_json(moved / 'artifacts' / 'current.json')['baseline']
    assert resolve_resource(moved / 'artifacts', baseline['model_path']).read_bytes() == b'local-model'
    assert resolve_resource(moved / 'artifacts', baseline['data_dir']).is_dir()


def test_legacy_absolute_resources_still_readable(tmp_path):
    source = tmp_path / 'model.gguf'
    assert resolve_resource(tmp_path / 'artifacts', str(source)) == source

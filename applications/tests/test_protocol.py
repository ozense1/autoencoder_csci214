"""Evaluation-protocol checks: development split and Matryoshka baseline."""
import numpy as np
import pytest

from config import AE_METHODS, DATA_DIR, FIXED_METHODS, MATRYOSHKA, METHODS
from data_utils import load_dataset, portable_path, read_json
from embeddings import normalize_rows
from matryoshka import build_matryoshka, truncate
from retrieval import Retriever, evaluate
from storage import build_collection, client, readiness, save_registry
from test_pipeline import small_dataset


# ---------- development split ----------

def test_default_split_is_test_and_dev_reads_train_tsv(tmp_path):
    small_dataset(tmp_path)
    (tmp_path / 'qrels' / 'train.tsv').write_text('query-id\tcorpus-id\tscore\nq\t1\t1\nq\t2\t1\n')
    _, _, test = load_dataset(tmp_path)
    _, _, dev = load_dataset(tmp_path, split='dev')
    assert test == {'q': {'0': 1.}}
    assert dev == {'q': {'1': 1., '2': 1.}}
    with pytest.raises(ValueError, match='Unknown qrels split'):
        load_dataset(tmp_path, split='validation')


def test_dev_split_file_only_required_when_requested(tmp_path):
    small_dataset(tmp_path)  # writes only qrels/test.tsv
    load_dataset(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path, split='dev')


@pytest.mark.skipif(not (DATA_DIR / 'qrels' / 'train.tsv').exists(), reason='bundled SciFact data not present')
def test_real_dev_split_counts_and_disjoint_from_test():
    _, _, dev = load_dataset(DATA_DIR, split='dev')
    _, _, test = load_dataset(DATA_DIR, split='test')
    assert len(dev) == 809
    assert sum(len(v) for v in dev.values()) == 919
    assert not set(dev) & set(test)


# ---------- Matryoshka baseline ----------

def test_matryoshka_is_fixed_not_trained():
    assert set(MATRYOSHKA) <= set(METHODS)
    assert set(MATRYOSHKA) <= set(FIXED_METHODS)
    assert not set(MATRYOSHKA) & set(AE_METHODS)


def test_truncate_hand_calculated():
    # Prefix [3, 4] has norm 5, so the result is [0.6, 0.8]; the tail is ignored.
    np.testing.assert_allclose(truncate([[3., 4., 12., -7.]], 2), [[.6, .8]], rtol=1e-6)
    out = truncate(np.random.default_rng(0).normal(size=(5, 8)), 4)
    assert out.shape == (5, 4)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1., rtol=1e-6)
    for bad in (0, 8, 9):
        with pytest.raises(ValueError):
            truncate(np.ones((2, 8)), bad)


def _fixture(tmp_path, n=40, dim=8):
    data = tmp_path / 'data'
    docs = small_dataset(data, n=n)
    vectors = normalize_rows(np.random.default_rng(7).normal(size=(n, dim)).astype(np.float32))
    np.save(tmp_path / 'vectors.npy', vectors)
    base = {'base_fingerprint': 'a' * 64, 'index_fingerprint': 'a' * 64, 'dimension': dim,
            'embedding_file': 'vectors.npy', 'data_dir': portable_path(data, tmp_path),
            'document_count': n, 'collection': 'baseline-test'}
    build_collection(tmp_path, base['collection'], docs, vectors, base['index_fingerprint'], lambda _: None)
    save_registry(tmp_path, {'baseline': base, 'variants': {}})
    return base, vectors


def test_matryoshka_index_query_path_and_reuse(tmp_path):
    base, vectors = _fixture(tmp_path)
    entry = build_matryoshka('matryoshka_128', 4, base, tmp_path, progress=lambda _: None)
    assert readiness(tmp_path)['matryoshka_128']['ready']
    # Documents in the index are exactly the truncated, renormalized vectors.
    stored = client(tmp_path).get_collection(entry['collection'], embedding_function=None).get(
        ids=['0', '5'], include=['embeddings'])
    expected = truncate(vectors, 4)
    for doc_id, emb in zip(stored['ids'], stored['embeddings']):
        np.testing.assert_allclose(emb, expected[int(doc_id)], rtol=1e-5, atol=1e-6)
    # A query equal to document 0 retrieves document 0 at cosine 1 (same transform).
    rows, _ = Retriever(tmp_path).search(vectors[0], 'matryoshka_128', entry, top_k=5)
    assert rows[0]['id'] == '0' and rows[0]['cosine'] == pytest.approx(1., abs=1e-4)
    assert len(rows) == 5
    again = build_matryoshka('matryoshka_128', 4, base, tmp_path, progress=lambda _: None)
    assert again == entry


# ---------- split-separated evaluation reports ----------

def test_dev_and_test_reports_are_kept_apart(tmp_path):
    base, vectors = _fixture(tmp_path)
    entry = build_matryoshka('matryoshka_128', 4, base, tmp_path, progress=lambda _: None)

    class Fake:
        def encode(self, prompt):
            return vectors[0], False

    entries = {'baseline': base, 'matryoshka_128': entry}
    queries, qrels = {'q': 'Test query'}, {'q': {'0': 1.}}
    dev = evaluate(['baseline', 'matryoshka_128'], entries, base, tmp_path, queries, qrels, Fake(), split='dev')
    test = evaluate(['baseline', 'matryoshka_128'], entries, base, tmp_path, queries, qrels, Fake())
    assert dev['split'] == 'dev' and test['split'] == 'test'
    assert dev['evaluation_fingerprint'] != test['evaluation_fingerprint']
    assert read_json(tmp_path / 'evaluations' / 'dev' / f"{dev['evaluation_fingerprint']}.json")['split'] == 'dev'
    assert [p.name for p in (tmp_path / 'evaluations').glob('*.json')] == [f"{test['evaluation_fingerprint']}.json"]
    for report in (dev, test):
        assert report['methods']['baseline']['metrics']['MRR@100'] == pytest.approx(1.)
        assert report['methods']['matryoshka_128']['metrics']['MRR@100'] == pytest.approx(1.)

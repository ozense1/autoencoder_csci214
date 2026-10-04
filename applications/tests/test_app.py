from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest
import streamlit as st

import embeddings
from data_utils import fingerprint, portable_path
from storage import build_collection, save_registry
from training import train_variant
from test_pipeline import small_dataset

APP = Path(__file__).resolve().parents[1] / 'app.py'


@pytest.fixture
def ui_environment(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    artifacts = tmp_path / 'artifacts'
    artifacts.mkdir()
    documents = small_dataset(data, n=20)
    model = tmp_path / 'model.gguf'
    model.write_bytes(b'fake-model-for-ui-test')
    monkeypatch.setenv('CAE_DATA_DIR', str(data))
    monkeypatch.setenv('CAE_ARTIFACT_DIR', str(artifacts))
    monkeypatch.setenv('CAE_MODEL_PATH', str(model))
    class FakeEmbedder:
        def __init__(self, *args, **kwargs):
            pass
        def encode(self, text):
            return np.array([1., 0., 0., 0., 0., 0., 0., 0.], dtype=np.float32), False
    monkeypatch.setattr(embeddings, 'Embedder', FakeEmbedder)
    monkeypatch.setattr(embeddings, 'embedding_spec', lambda *a: ({'fake': True}, 'a' * 64))
    yield data, artifacts, documents, model
    st.cache_resource.clear()
    st.cache_data.clear()


def ready_environment(env):
    data, artifacts, documents, model = env
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(len(documents), 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    np.save(artifacts / 'documents.npy', vectors)
    entry = {'base_fingerprint': 'a' * 64, 'index_fingerprint': 'a' * 64,
             'collection': 'ui-baseline', 'dimension': 8, 'document_count': len(documents),
             'data_dir': portable_path(data, artifacts), 'model_path': portable_path(model, artifacts), 'embedding_file': 'documents.npy',
             'runtime': {'threads': 1, 'gpu_layers': 0}, 'truncated_documents': 0}
    build_collection(artifacts, entry['collection'], documents, vectors, entry['index_fingerprint'], lambda _: None)
    save_registry(artifacts, {'baseline': entry, 'variants': {}})
    train_variant('mse', entry, artifacts, epochs=1, latent_dim=3, threads=1, progress=lambda _: None)


def test_missing_artifacts_shows_preparation_command(ui_environment):
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    assert any('Prepare the document embeddings' in item.value for item in at.info)
    assert any('train_autoencoder.py' in item.value and 'prepare' in item.value for item in at.code)


def test_single_compare_free_text_and_missing_variant(ui_environment):
    ready_environment(ui_environment)
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    at.button(key='search').click().run()
    assert not at.exception and not at.error
    assert 'baseline' in at.session_state['search_result']['methods']
    at.radio(key='view').set_value('Compare methods').run()
    at.multiselect(key='compare_methods').set_value(['baseline', 'mse']).run()
    at.button(key='search').click().run()
    assert not at.exception and not at.error
    assert set(at.session_state['search_result']['methods']) == {'baseline', 'mse'}
    at.radio(key='query_source').set_value('Free text').run()
    at.text_area(key='query_text').set_value('Does this retrieve documents?').run()
    at.button(key='search').click().run()
    assert not at.exception and not at.error
    assert any('No relevance judgments' in caption.value for caption in at.caption)
    at.radio(key='view').set_value('Single method').run()
    at.selectbox(key='method').set_value('mse_cosine').run()
    assert at.button(key='search').disabled
    assert any('--variant mse_cosine' in item.value for item in at.code)


def test_full_evaluation_and_training_chart(ui_environment):
    ready_environment(ui_environment)
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    at.button(key='evaluate').click().run()
    assert not at.exception and not at.error
    report = at.session_state['evaluation_report']
    assert report['query_count'] == 1
    assert set(report['methods']) == {'baseline', 'mse'}
    assert any('nDCG@10' in table.value.columns for table in at.dataframe)
    assert at.selectbox(key='training_method').value == 'mse'
    assert len(list((ui_environment[1] / 'evaluations').glob('*.json'))) == 1

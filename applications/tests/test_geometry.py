"""Checks for the similarity-preservation (geometry) term and the split seed."""
import math

import numpy as np
import pytest
import torch

from autoencoder import Autoencoder, coefficients, geometry, latent_statistics, loss_terms
from data_utils import portable_path, read_json
from embeddings import normalize_rows
from experiment import summarize
from storage import build_collection, save_registry
from test_pipeline import small_dataset
from training import train_variant


def test_toy_example_from_handout():
    # Input pair cosine 0.8, latent pair cosine 0.3: squared error 0.25 for (1,2)
    # and (2,1); the mean over B(B-1)=2 off-diagonal entries is 0.25.
    x = torch.tensor([[1., 0.], [.8, .6]])
    z = torch.tensor([[1., 0., 0.], [.3, math.sqrt(1 - .09), 0.]])
    assert geometry(x, z).item() == pytest.approx(.25, abs=1e-6)


def test_three_document_hand_calculation():
    # Pairs (cos_x, cos_z): (0, 0), (0, 1), (1, 0)  ->  errors 0, 1, 1 -> mean 2/3.
    x = torch.tensor([[1., 0.], [0., 1.], [0., 1.]])
    z = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
    assert geometry(x, z).item() == pytest.approx(2 / 3, abs=1e-6)


def test_identical_codes_are_penalized_and_scale_is_ignored():
    torch.manual_seed(0)
    x = torch.randn(6, 10)
    z = torch.randn(6, 4)
    collapsed = torch.ones(6, 4) * .3
    xn = torch.nn.functional.normalize(x, dim=1)
    expected = ((xn @ xn.T - 1)[~torch.eye(6, dtype=torch.bool)]).square().mean()
    torch.testing.assert_close(geometry(x, collapsed), expected)
    assert geometry(x, collapsed) > .5  # random directions: cosines near 0, so error near 1
    torch.testing.assert_close(geometry(x, 5 * z), geometry(x, z))
    torch.testing.assert_close(geometry(x, .01 * z), geometry(x, z))


def test_single_example_and_finite_for_zero_codes():
    assert geometry(torch.randn(1, 5), torch.randn(1, 3)).item() == 0.
    assert torch.isfinite(geometry(torch.randn(4, 5), torch.zeros(4, 3)))


def test_geometry_gradients_reach_encoder():
    torch.manual_seed(1)
    model = Autoencoder(8, 3)
    x = torch.randn(16, 8)
    z, _ = model(x)
    geometry(x, z).backward()
    assert model.encoder.weight.grad.abs().sum() > 0
    assert model.encoder.bias.grad.abs().sum() > 0
    assert model.decoder_bias.grad is None  # decoder is not involved in this term


def test_coefficients_keep_original_methods_unchanged():
    for method in ('mse', 'mse_cosine', 'mse_contractive', 'mse_cosine_contractive'):
        assert 'gamma' not in coefficients(method, gamma=5.)
    assert coefficients('mse_cosine_geometry', gamma=.5) == {'alpha': 1., 'beta': .1, 'lam': 0., 'gamma': .5}


def test_total_includes_geometry_only_when_weighted():
    torch.manual_seed(2)
    model = Autoencoder(6, 3)
    x = torch.randn(5, 6)
    terms = loss_terms(model, x, coefficients('mse_cosine_geometry', gamma=2.))
    torch.testing.assert_close(terms['total'], terms['mse'] + .1 * terms['cosine'] + 2. * terms['geometry'])
    plain = loss_terms(model, x, coefficients('mse_cosine'))
    torch.testing.assert_close(plain['total'], plain['mse'] + .1 * plain['cosine'])
    assert plain['geometry'] > 0  # still logged as a diagnostic


def test_latent_statistics_detect_collapse():
    model = Autoencoder(6, 3)
    with torch.no_grad():
        model.encoder.weight.zero_()
        model.encoder.bias.fill_(.5)
    stats = latent_statistics(model, torch.randn(20, 6))
    assert stats['coordinates_std_below_1e-3'] == 3
    assert stats['mean_pairwise_cosine'] == pytest.approx(1., abs=1e-5)


def _setup(tmp_path):
    data = tmp_path / 'data'
    docs = small_dataset(data)
    vectors = normalize_rows(np.random.default_rng(7).normal(size=(40, 8)).astype(np.float32))
    np.save(tmp_path / 'vectors.npy', vectors)
    base = {'base_fingerprint': 'a' * 64, 'dimension': 8, 'embedding_file': 'vectors.npy',
            'data_dir': portable_path(data, tmp_path), 'document_count': 40,
            'collection': 'baseline-test', 'index_fingerprint': 'a' * 64}
    build_collection(tmp_path, base['collection'], docs, vectors, base['index_fingerprint'], lambda _: None)
    save_registry(tmp_path, {'baseline': base, 'variants': {}})
    return base


def test_gamma_zero_ablation_reproduces_mse_cosine(tmp_path):
    base = _setup(tmp_path)
    quiet = dict(latent_dim=3, epochs=3, threads=1, progress=lambda _: None)
    reference = train_variant('mse_cosine', base, tmp_path, **quiet)
    ablation = train_variant('mse_cosine_geometry', base, tmp_path, gamma=0., **quiet)
    assert ablation['training_fingerprint'] != reference['training_fingerprint']
    a = np.load((tmp_path / reference['checkpoint']).parent / 'document_latents.npy')
    b = np.load((tmp_path / ablation['checkpoint']).parent / 'document_latents.npy')
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)
    assert 'latent_stats' in ablation and ablation['parameter_count'] == 8 * 3 + 3 + 8
    history = read_json(tmp_path / ablation['history'])
    assert 'validation_geometry' in history[0] and 'train_geometry' in history[0]


def test_split_seed_fixes_split_but_not_initialization(tmp_path):
    base = _setup(tmp_path)
    quiet = dict(latent_dim=3, epochs=1, threads=1, progress=lambda _: None)
    first = train_variant('mse', base, tmp_path, seed=1, split_seed=7, **quiet)
    second = train_variant('mse', base, tmp_path, seed=2, split_seed=7, **quiet)
    split = lambda e: read_json((tmp_path / e['checkpoint']).parent / 'split.json')
    assert split(first) == split(second)
    assert first['checkpoint_sha256'] != second['checkpoint_sha256']
    assert first['config']['split_seed'] == 7
    legacy = train_variant('mse', base, tmp_path, seed=7, **quiet)
    assert 'split_seed' not in legacy['config']  # original fingerprints unchanged


def test_summary_mean_and_std():
    rows = [{'label': 'a', 'dimension': 128, **{m: v for m in ('nDCG@10', 'Recall@10', 'Recall@100', 'MAP@100', 'MRR@100')}}
            for v in (.5, .7)]
    item = summarize(rows)[0]
    assert item['runs'] == 2 and item['nDCG@10'] == pytest.approx(.6)
    assert item['nDCG@10_std'] == pytest.approx(math.sqrt(.02))

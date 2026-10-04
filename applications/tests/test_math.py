import math
import pytest
import torch

from autoencoder import Autoencoder, coefficients, loss_terms
from metrics import mean_metrics, query_metrics


def test_analytic_penalty_and_parameter_gradients_equal_autograd():
    torch.manual_seed(1)
    model = Autoencoder(5, 3).double()
    x = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
    z = model.encode(x)
    rows = [torch.autograd.grad(z[:, i].sum(), x, create_graph=True, retain_graph=True)[0] for i in range(3)]
    exact = torch.stack(rows, dim=1).square().sum((1, 2)).mean()
    analytic = model.contractive(z)
    torch.testing.assert_close(analytic, exact)
    parameters = (model.encoder.weight, model.encoder.bias)
    expected = torch.autograd.grad(exact, parameters, retain_graph=True)
    actual = torch.autograd.grad(analytic, parameters)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize('method,beta,lam', [
    ('mse', 0., 0.), ('mse_cosine', .1, 0.), ('mse_contractive', 0., .01), ('mse_cosine_contractive', .1, .01),
])
def test_variant_weights_and_training_gradients(method, beta, lam):
    model = Autoencoder(5, 3)
    terms = loss_terms(model, torch.randn(4, 5), coefficients(method))
    torch.testing.assert_close(terms['total'], terms['mse'] + beta * terms['cosine'] + lam * terms['contractive'])
    terms['total'].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_metrics_hand_calculated():
    result = query_metrics(['a', 'x', 'b'], {'a': 1, 'b': 1, 'missing': 1})
    assert result['Recall@10'] == result['Recall@100'] == pytest.approx(2 / 3)
    assert result['MAP@100'] == pytest.approx((1 + 2 / 3) / 3)
    assert result['MRR@100'] == 1.
    assert result['nDCG@10'] == pytest.approx((1 + .5) / (1 + 1 / math.log2(3) + .5))
    assert mean_metrics([result, result]) == result


def test_metric_cutoffs_and_empty_results():
    ranking = [f'x{i}' for i in range(100)] + ['a']
    assert all(value == 0. for value in query_metrics(ranking, {'a': 1}).values())
    assert all(value == 0. for value in query_metrics([], {'a': 1}).values())
    assert all(value == 0. for value in query_metrics(['a'], {}).values())
    with pytest.raises(ValueError, match='unique'):
        query_metrics(['a', 'a'], {'a': 1})


def test_graded_ndcg_and_ap_denominator():
    values = query_metrics(['b', 'a'], {'a': 2, 'b': 1})
    assert values['nDCG@10'] == pytest.approx((1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3)))
    assert values['MAP@100'] == 1.

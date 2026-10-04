"""A shallow tied-weight autoencoder with an exact analytic Jacobian penalty."""
import torch
from torch import nn
from torch.nn import functional as F


class Autoencoder(nn.Module):
    def __init__(self, input_dim=768, latent_dim=128):
        super().__init__()
        if not 0 < latent_dim < input_dim:
            raise ValueError('Latent dimension must be positive and smaller than the input.')
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))

    def encode(self, x):
        return torch.tanh(self.encoder(x))

    def forward(self, x):
        z = self.encode(x)
        reconstructed = F.linear(z, self.encoder.weight.T, self.decoder_bias)
        return z, reconstructed

    def contractive(self, z):
        # J[b,i,j] = (1-z[b,i]^2) * W[i,j]. Sum over i,j, then mean over b.
        # No detached tensors: gradients include the activation AND weight terms.
        derivative_squared = (1 - z.square()).square()
        row_norm_squared = self.encoder.weight.square().sum(dim=1)
        return (derivative_squared * row_norm_squared).sum(dim=1).mean()


def coefficients(method, alpha=1., beta=.1, lam=.01):
    if method not in ('mse', 'mse_cosine', 'mse_contractive', 'mse_cosine_contractive'):
        raise ValueError(f'Unknown AE method: {method}')
    return {'alpha': alpha, 'beta': beta if 'cosine' in method else 0., 'lam': lam if 'contractive' in method else 0.}


def loss_terms(model, x, weights):
    z, reconstructed = model(x)
    mse = F.mse_loss(reconstructed, x)
    cosine = (1 - F.cosine_similarity(reconstructed, x, dim=1, eps=1e-8)).mean()
    contractive = model.contractive(z)
    total = weights['alpha'] * mse + weights['beta'] * cosine + weights['lam'] * contractive
    return {'total': total, 'mse': mse, 'cosine': cosine, 'contractive': contractive}

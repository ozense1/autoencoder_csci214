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

AE_LOSSES = ('mse', 'mse_cosine', 'mse_contractive', 'mse_cosine_contractive', 'mse_cosine_geometry')
GEOMETRY_VERSION = 'offdiag-cosine-sq-uniform-v1'


def coefficients(method, alpha=1., beta=.1, lam=.01, gamma=1.):
    if method not in AE_LOSSES:
        raise ValueError(f'Unknown AE method: {method}')
    weights = {'alpha': alpha, 'beta': beta if 'cosine' in method else 0., 'lam': lam if 'contractive' in method else 0.}
    # 'gamma' is only added for the geometry method, so the training fingerprints
    # (and cached checkpoints) of the four original methods stay unchanged.
    if 'geometry' in method:
        weights['gamma'] = gamma
    return weights


def geometry(x, z, eps=1e-8):
    """Pairwise similarity preservation within a batch.

    S^x_ij = cos(x_i, x_j) in the 768-d input space, S^z_ij = cos(z_i, z_j) in the
    latent space. Returns mean over the B(B-1) off-diagonal pairs of (S^x - S^z)^2.
    Cosine is computed on normalized vectors (the geometry used at retrieval), so
    rescaling z cannot change the penalty; identical codes give S^z = 1 everywhere
    and are penalized whenever the inputs differ. The input side carries no
    trainable parameters, so gradients reach the encoder only through z.
    """
    size = x.shape[0]
    if size < 2:
        return z.new_zeros(())
    xn = F.normalize(x, dim=1, eps=eps)
    zn = F.normalize(z, dim=1, eps=eps)
    difference = xn @ xn.T - zn @ zn.T
    off_diagonal = ~torch.eye(size, dtype=torch.bool, device=z.device)
    return difference[off_diagonal].square().mean()


def loss_terms(model, x, weights):
    z, reconstructed = model(x)
    mse = F.mse_loss(reconstructed, x)
    cosine = (1 - F.cosine_similarity(reconstructed, x, dim=1, eps=1e-8)).mean()
    contractive = model.contractive(z)
    # Always computed and logged (a diagnostic for every method); it only trains
    # the encoder when gamma > 0.
    geo = geometry(x, z)
    total = weights['alpha'] * mse + weights['beta'] * cosine + weights['lam'] * contractive
    if 'gamma' in weights:
        total = total + weights['gamma'] * geo
    return {'total': total, 'mse': mse, 'cosine': cosine, 'contractive': contractive, 'geometry': geo}


@torch.no_grad()
def latent_statistics(model, vectors, batch=512):
    """Collapse diagnostics on raw (pre-normalization) z over a set of documents."""
    z = torch.cat([model.encode(chunk) for chunk in vectors.split(batch)])
    std = z.std(dim=0)
    return {
        'mean_norm': z.norm(dim=1).mean().item(),
        'mean_coordinate_std': std.mean().item(),
        'min_coordinate_std': std.min().item(),
        'coordinates_std_below_1e-3': int((std < 1e-3).sum().item()),
        'saturated_fraction': (z.abs() > .99).float().mean().item(),
        'mean_pairwise_cosine': geometry_mean_cosine(z),
    }


def geometry_mean_cosine(z, sample=2000):
    zn = F.normalize(z[:sample], dim=1, eps=1e-8)
    similarity = zn @ zn.T
    size = similarity.shape[0]
    return ((similarity.sum() - similarity.diagonal().sum()) / max(size * (size - 1), 1)).item()

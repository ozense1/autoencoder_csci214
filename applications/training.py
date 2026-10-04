"""Fair, reproducible training on document embeddings, never on qrels."""
import os
from pathlib import Path
import time

import numpy as np
import torch
from chromadb.errors import NotFoundError

from autoencoder import GEOMETRY_VERSION, Autoencoder, coefficients, latent_statistics, loss_terms
from data_utils import fingerprint, load_dataset, read_json, resolve_resource, sha256_file, write_json
from embeddings import normalize_rows
from storage import build_collection, load_registry, save_registry


def train_variant(method, baseline, artifact_dir, *, latent_dim=128, epochs=100, batch_size=64,
                    lr=.001, alpha=1., beta=.1, lam=.01, gamma=1., seed=7, split_seed=None,
                    patience=10, device='cpu', threads=4, progress=print):
    """Train one variant. seed controls initialization and minibatch order;
    split_seed (default: same as seed) controls the 90/10 document split."""
    artifact_dir = Path(artifact_dir)
    weights = coefficients(method, alpha, beta, lam, gamma)
    config = {
        'base_fingerprint': baseline['base_fingerprint'], 'method': method,
        'input_dim': baseline['dimension'], 'latent_dim': latent_dim,
        'architecture': 'tied-shallow-tanh-linear-decoder-v1', 'weights': weights,
        'epochs': epochs, 'batch_size': batch_size, 'lr': lr, 'seed': seed,
        'patience': patience, 'min_delta': 1e-8, 'train_fraction': .9,
        'torch_version': str(torch.__version__), 'device': device, 'threads': threads,
    }
    # Optional keys are added only when used, so earlier fingerprints stay valid.
    if 'gamma' in weights:
        config['loss_version'] = GEOMETRY_VERSION
    if split_seed is not None:
        config['split_seed'] = split_seed
    config_fp = fingerprint(config)
    folder = artifact_dir / 'models' / f'{method}-{config_fp[:16]}'
    entry_path = folder / 'entry.json'
    registry = load_registry(artifact_dir)
    if entry_path.exists():
        cached = read_json(entry_path)
        candidate = dict(registry)
        candidate['variants'] = {**registry.get('variants', {}), method: cached}
        # Check before registering; never publish an incomplete checkpoint/index.
        try:
            from storage import client
            collection = client(artifact_dir).get_collection(cached['collection'], embedding_function=None)
            if (cached['training_fingerprint'] == config_fp
                and sha256_file(artifact_dir / cached['checkpoint']) == cached['checkpoint_sha256']
                and collection.count() == baseline['document_count']
                and collection.metadata.get('fingerprint') == cached['index_fingerprint']):
                save_registry(artifact_dir, candidate)
                progress(f'{method}: reusing completed training run.')
                return cached
        except (FileNotFoundError, ValueError, NotFoundError):
            pass
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    all_vectors = torch.from_numpy(np.array(np.load(artifact_dir / baseline['embedding_file']), copy=True))
    split_generator = torch.Generator().manual_seed(seed if split_seed is None else split_seed)
    permutation = torch.randperm(len(all_vectors), generator=split_generator)
    split = int(.9 * len(all_vectors))
    train_indices, validation_indices = permutation[:split], permutation[split:]
    training = all_vectors[train_indices].to(device)
    validation = all_vectors[validation_indices].to(device)
    model = Autoencoder(baseline['dimension'], latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    order_generator = torch.Generator().manual_seed(seed + 1)
    write_json(folder / 'config.json', config)
    write_json(folder / 'split.json', {'training_indices': train_indices.tolist(), 'validation_indices': validation_indices.tolist()})
    history = []
    best_loss, best_epoch, stale, best_state = float('inf'), 0, 0, None
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        sums = {}
        order = torch.randperm(len(training), generator=order_generator)
        for indices in order.split(batch_size):
            batch = training[indices.to(device)]
            optimizer.zero_grad(set_to_none=True)
            terms = loss_terms(model, batch, weights)
            if not torch.isfinite(terms['total']):
                raise RuntimeError(f'Non-finite {method} training loss; adjust settings.')
            terms['total'].backward()
            optimizer.step()
            for key, value in terms.items():
                sums[key] = sums.get(key, 0.) + value.detach().item() * len(batch)
        model.eval()
        # The analytic penalty works in no_grad during validation.
        with torch.no_grad():
            valid = {key: value.item() for key, value in loss_terms(model, validation, weights).items()}
        if not all(np.isfinite(value) for value in valid.values()):
            raise RuntimeError('Non-finite validation loss.')
        row = {'epoch': epoch, **{f'train_{key}': value / len(training) for key, value in sums.items()},
               **{f'validation_{key}': value for key, value in valid.items()}}
        history.append(row)
        if valid['total'] < best_loss - 1e-8:
            best_loss, best_epoch, stale = valid['total'], epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            terms = ', '.join(f'{k}={v:.3g}' for k, v in valid.items() if k != 'total')
            progress(f'{method}: epoch {epoch}/{epochs}, validation={valid["total"]:.6f} ({terms}), best={best_epoch}, elapsed={time.perf_counter()-started:.0f}s')
            write_json(folder / 'history.json', history)
        if stale >= patience:
            progress(f'{method}: early stop at epoch {epoch}.')
            break
    checkpoint = folder / 'checkpoint.pt'
    temporary = folder / 'checkpoint.tmp'
    torch.save({'state_dict': best_state, 'config': config, 'best_epoch': best_epoch}, temporary)
    os.replace(temporary, checkpoint)
    write_json(folder / 'history.json', history)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        latent = np.concatenate([model.encode(batch.to(device)).cpu().numpy() for batch in all_vectors.split(512)])
    stats = latent_statistics(model, all_vectors.to(device))
    latent = normalize_rows(latent)
    np.save(folder / 'document_latents.npy', latent)
    checksum = sha256_file(checkpoint)
    index_fp = fingerprint({'base': baseline['base_fingerprint'], 'checkpoint': checksum, 'normalize_z': True})
    collection = f'scifact-{method.replace("_", "-")}-{index_fp[:16]}'
    documents, _, _ = load_dataset(resolve_resource(artifact_dir, baseline['data_dir']))
    build_collection(artifact_dir, collection, documents, latent, index_fp, progress)
    entry = {
        'base_fingerprint': baseline['base_fingerprint'], 'index_fingerprint': index_fp,
        'training_fingerprint': config_fp, 'checkpoint_sha256': checksum,
        'checkpoint': str(checkpoint.relative_to(artifact_dir)),
        'history': str((folder / 'history.json').relative_to(artifact_dir)),
        'collection': collection, 'dimension': latent_dim, 'document_count': len(documents),
        'config': config, 'best_epoch': best_epoch, 'best_validation_loss': best_loss,
        'epochs_completed': len(history), 'training_seconds': time.perf_counter() - started,
        'latent_stats': stats, 'parameter_count': sum(p.numel() for p in model.parameters()),
    }
    write_json(entry_path, entry)
    registry = load_registry(artifact_dir)
    if registry['baseline']['base_fingerprint'] != baseline['base_fingerprint']:
        raise RuntimeError('Baseline changed while training; rerun against the current baseline.')
    registry['variants'][method] = entry
    save_registry(artifact_dir, registry)
    progress(f'{method}: ready at {latent_dim} dimensions; best epoch {best_epoch}.')
    return entry


def load_encoder(artifact_dir, entry):
    checkpoint = Path(artifact_dir) / entry['checkpoint']
    if sha256_file(checkpoint) != entry['checkpoint_sha256']:
        raise ValueError('Checkpoint content changed. Rebuild the corresponding index.')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if fingerprint(saved['config']) != entry['training_fingerprint']:
        raise ValueError('Checkpoint configuration does not match the index.')
    config = saved['config']
    model = Autoencoder(config['input_dim'], config['latent_dim'])
    model.load_state_dict(saved['state_dict'])
    model.eval()
    return model

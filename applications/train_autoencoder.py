"""Prepare SciFact retrieval and train independent autoencoder variants."""
import argparse
import math
from pathlib import Path

from filelock import FileLock, Timeout

from config import AE_METHODS, ARTIFACT_DIR, DATA_DIR, DEFAULT_MODEL
from prepare import prepare
from training import train_variant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'train'])
    parser.add_argument('--variant', choices=['all', *AE_METHODS], default='all')
    parser.add_argument('--model-path', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--artifact-dir', type=Path, default=ARTIFACT_DIR)
    parser.add_argument('--latent-dim', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--alpha', type=float, default=1.)
    parser.add_argument('--beta', type=float, default=.1)
    parser.add_argument('--lambda', dest='lam', type=float, default=.01)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--gpu-layers', type=int, default=0, help='Optional llama.cpp GPU offload; -1 means all layers.')
    args = parser.parse_args()
    if not 0 < args.latent_dim < 768:
        parser.error('--latent-dim must be between 1 and 767')
    if any(v < 1 for v in (args.epochs, args.batch_size, args.patience, args.threads)):
        parser.error('Epochs, batch size, patience and threads must be positive.')
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error('--lr must be finite and positive.')
    if not all(math.isfinite(v) and v >= 0 for v in (args.alpha, args.beta, args.lam)) or args.alpha == 0:
        parser.error('Weights must be finite and nonnegative; alpha must be positive.')
    args.artifact_dir = args.artifact_dir.resolve()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(args.artifact_dir / 'pipeline.lock'), timeout=0):
            baseline = prepare(args.model_path, args.data_dir, args.artifact_dir, args.threads, args.gpu_layers)
            if args.command == 'train':
                for variant in AE_METHODS if args.variant == 'all' else [args.variant]:
                    train_variant(variant, baseline, args.artifact_dir,
                                  **{key: getattr(args, key) for key in ('latent_dim', 'epochs', 'batch_size', 'lr', 'alpha', 'beta', 'lam', 'seed', 'patience', 'device', 'threads')})
    except Timeout:
        parser.exit(1, 'Another preparation/training process is active. Wait for it to finish.\n')


if __name__ == '__main__':
    main()

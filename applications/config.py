"""Shared defaults and readable names for the retrieval methods."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data' / 'scifact'
ARTIFACT_DIR = ROOT / 'artifacts'
MODEL_DIR = ROOT / 'models'
DEFAULT_MODEL = MODEL_DIR / 'embeddinggemma-300m-Q4_0.gguf'
METHODS = {
    'baseline': 'Embedding only',
    'matryoshka_128': 'Embedding: Matryoshka first 128',
    'mse': 'Embedding + AE: MSE',
    'mse_cosine': 'Embedding + AE: MSE + Cosine',
    'mse_cosine_contractive': 'Embedding + AE: MSE + Cosine + Contractive',
    'mse_contractive': 'Embedding + AE: MSE + Contractive',
    'mse_cosine_geometry': 'Embedding + AE: MSE + Cosine + Geometry',
}

# Fixed (untrained) compression baselines: keep the first k coordinates, then
# L2-normalize again, identically for queries and documents.
MATRYOSHKA = {'matryoshka_128': 128}
FIXED_METHODS = ('baseline', *MATRYOSHKA)
AE_METHODS = tuple(k for k in METHODS if k not in FIXED_METHODS)

# Retrieval development split (lab protocol): qrels/train.tsv selects
# hyperparameters qrels/test.tsv is reserved for final evaluation.
QREL_SPLITS = {'test': 'test.tsv', 'dev': 'train.tsv'}
DIMENSION = 768
CONTEXT = 2048
PROMPT_VERSION = 'embeddinggemma-retrieval-v1'
